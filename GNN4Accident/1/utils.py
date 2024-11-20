from .models import *

import argparse
parser = argparse.ArgumentParser(description='参数解析示例')

#添加参数
parser.add_argument('-d', type=int, default=16, help='超参数d的值')
parser.add_argument('-p', type=float, default=0.5, help='超参数p的值')
parser.add_argument('--class_num', type=int, default=8, help='类别数量')
parser.add_argument('--num_epochs', type=int, default=301, help='训练周期')
parser.add_argument('--file_path', type=str, default='exp-severity/', help='文件路径')

# 解析命令行参数
args = parser.parse_args()

# 使用解析后的参数
d = args.d
p = args.p
class_num = args.class_num
num_epochs = args.num_epochs
file_path = args.file_path

# 获取颜色映射
cmap = cm.get_cmap('cool', class_num)

# 存储结果的列表
all_res = []

from .dataloader import *
def draw_with_labels(df_nodes, pos_dict, model_name='test'):
    fig, ax = plt.subplots(figsize=(6, 5))
    for i in range(class_num):
        G = nx.MultiGraph()
        G.add_nodes_from(df_nodes[df_nodes['label'] == i].index)
        nx.draw(G, pos=pos_dict, node_color=colors.rgb2hex(cmap(i)), node_size=3, label=i)

    norm = colors.BoundaryNorm(boundaries=range(8), ncolors=8)
    cs = cm.ScalarMappable(norm=norm, cmap=cmap)
    plt.colorbar(cs, ax=ax, orientation='vertical', label='Severity')
    plt.title(model_name, y=-0.01)
    plt.show()

def grapg_check(e): # e是一个元组 e.g. ('city', 'state'):
    city_name, state_abbrev = e[0].lower().replace(" ", "_"), us_state_to_abbrev[e[1]].lower()
    city_format = e[0]+' ('+us_state_to_abbrev[e[1]]+')'
    if os.path.exists(file_path+city_name+'_'+state_abbrev+'/processed'):
        shutil.rmtree(file_path+city_name+'_'+state_abbrev+'/processed')
    dataset = LoadDataset(file_path, city_name+'_'+state_abbrev)
    data = dataset[0]
    class_num = dataset.num_classes
    print(f'city name:{e}')
    print(f'Number of graphs: {len(dataset)}')
    print(f'Number of node features: {dataset.num_features}')
    print(f'Number of edge features: {dataset.num_edge_features}')
    print(f'Number of classes: {dataset.num_classes}')
    print(f'Number of nodes: {data.num_nodes}')
    print(f'Number of edges: {data.num_edges}')
    print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
    print(f'Contains isolated nodes: {data.has_isolated_nodes()}')
    print(f'Contains self-loops: {data.has_self_loops()}')
    print(f'Is undirected: {data.is_undirected()}')

# 互信息最大化损失
class MutualInformationLoss(nn.Module):
    def __init__(self):
        super(MutualInformationLoss, self).__init__()

    def forward(self, local_features, global_features):
        batch_size = local_features.size(0)
        scores = torch.matmul(local_features, global_features.T)
        labels = torch.arange(batch_size).cuda()
        loss = F.cross_entropy(scores, labels)
        return loss

# InfoNCE损失
class InfoNCELoss(nn.Module):
    def __init__(self, temperature=0.1):
        super(InfoNCELoss, self).__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        batch_size = features.size(0)
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float()
        contrast_feature = torch.cat(torch.unbind(features, dim=0), dim=0)
        anchor_feature = contrast_feature
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        mask = mask.repeat(batch_size, 1)
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * 2).view(-1, 1).cuda(),
            0
        )
        mask = mask * logits_mask
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        loss = - (self.temperature / 0.07) * mean_log_prob_pos
        loss = loss.view(batch_size, 2).mean()
        return loss