from .models import *
from pylab import cm
device = torch.device('cuda')
device, torch.cuda.current_device()


def train(model, data, optimizer, loss_fn=MutualInformationLoss):
    model.train()
    optimizer.zero_grad()
    loss = loss_fn(model()[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()


@torch.no_grad()
def test(model, data):
    model.eval()
    logits, measures = model().detach(), []
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        pred = logits[mask].max(1)[1]
        mea = f1_score(data.y[mask].cpu(), pred.cpu(), average='weighted')
        measures.append(mea)
    label_pred = logits.max(1)[1]

    mask = data.test_mask
    scores = logits[mask][:, 1]
    pred = logits[mask].max(1)[1]
    test_y = data.y[mask]

    test_acc = pred.eq(test_y).sum().item() / mask.sum().item()
    return measures, label_pred, test_acc


def train_loop(model, data, optimizer, num_epochs, pos_dict,model_name='', city_name=''):
    epochs, train_measures, valid_measures, test_measures, test_accs = [], [], [], [], []
    coords = data.coords.cpu().numpy()
    gdf_pred = pd.DataFrame({'x': coords[:, 0], 'y': coords[:, 1]})
    for epoch in range(num_epochs):
        train(model, data, optimizer)
        log = 'Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
        measures, label_pred, test_acc = test(model, data)
        train_mea, valid_mea, test_mea = measures
        epochs.append(epoch)
        train_measures.append(train_mea)
        valid_measures.append(valid_mea)
        test_measures.append(test_mea)
        test_accs.append(test_acc)

        if epoch % 5 == 0 and epoch != num_epochs - 1:
            clear_output(True)
            fig, (ax1, ax) = plt.subplots(1, 2, figsize=(30, 12))
            gdf_pred['label'] = label_pred.cpu().numpy()
            for i in range(class_num):
                G = nx.MultiGraph()
                G.add_nodes_from(gdf_pred[gdf_pred['label'] == i].index)
                sub1 = nx.draw(G, pos=pos_dict, ax=ax1, node_color=colors.rgb2hex(cmap(i)), node_size=10, label=i)

            ax.text(1, 1, log.format(epoch, train_measures[-1], valid_measures[-1], test_measures[-1]), fontsize=18)
            ax.plot(epochs, train_measures, "r", epochs, valid_measures, "g", epochs, test_measures, "b")
            ax.set_ylim([0, 1])
            ax.legend(["train", "valid", "test"])
            norm = colors.BoundaryNorm(boundaries=range(8), ncolors=8)
            plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax1, orientation='vertical', label='Severity')
            ax1.set_title(city_name + ' ' + model_name, y=-0.01)
            plt.show()
        if epoch == num_epochs - 1:
            clear_output(True)
            fig, ax1 = plt.subplots(figsize=(15, 12))  # 修改为只创建一个子图

            gdf_pred['label'] = label_pred.cpu().numpy()
            for i in range(class_num):
                G = nx.MultiGraph()
                G.add_nodes_from(gdf_pred[gdf_pred['label'] == i].index)
                nx.draw(G, pos=pos_dict, ax=ax1, node_color=colors.rgb2hex(cmap(i)), node_size=10, label=i)

            # 添加图例
            handles = [
                plt.Line2D([0], [0], marker='o', color='w', label=str(i), markerfacecolor=colors.rgb2hex(cmap(i)),
                           markersize=10) for i in range(class_num)]
            ax1.legend(handles=handles, title='Severity')

            # 添加颜色条
            norm = colors.BoundaryNorm(boundaries=range(class_num), ncolors=class_num)
            cbar = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax1, orientation='vertical',
                                label='Severity')
            cbar.set_ticks(range(class_num))
            cbar.set_ticklabels(range(class_num))

            # 设置图的标题
            ax1.set_title(city_name + ' ' + model_name, y=1.05)

            # 确保文件夹路径存在，如果不存在就创建它
            folder_path = 'severity_plt'
            os.makedirs(folder_path, exist_ok=True)

            # 构建完整的文件保存路径
            file_path = os.path.join(folder_path, f'{city_name}_{model_name}.png')

            # 保存图片到指定路径
            plt.savefig(file_path, format='png', dpi=500, bbox_inches='tight')
            plt.show()

    select_idx = np.argmax(valid_measures[num_epochs // 2:]) + num_epochs // 2
    final_test_mea = np.array(test_measures)[select_idx]
    final_test_acc = np.array(test_accs)[select_idx]
    print('F measure {:.5f} | Test Accuracy {:.5f}'.format(final_test_mea, final_test_acc))
    return (round(final_test_mea * 100, 2), round(final_test_acc * 100, 2))