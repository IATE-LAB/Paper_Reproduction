from .train import *
from .dataloader import LoadDataset, us_state_to_abbrev, train_test_split_stratify

torch.manual_seed(7)
plt.style.use("ggplot")
warnings.filterwarnings("ignore", category=UserWarning)
def main():

    for e in [('Miami', 'Florida'), ('Los Angeles', 'California'), ('Orlando', 'Florida'),
              ('Dallas', 'Texas'), ('Houston', 'Texas'), ('New York', 'New York'), ('Chicago', 'Illinois'),
              ('Las Vegas', 'Nevada'), ('San Francisco', 'California'), ('Washington', 'District of Columbia'),
              ('Seattle', 'Washington'),
              ('Boston', 'Massachusetts'), ('Philadelphia', 'Pennsylvania'), ('Phoenix', 'Arizona'),
              ('Atlanta', 'Georgia'),
              ('Denver', 'Colorado'), ('San Diego', 'California'), ('Portland', 'Oregon'), ('Nashville', 'Tennessee'),
              ('Charlotte', 'North Carolina'), ('Detroit', 'Michigan'), ('Indianapolis', 'Indiana'),
              ('Milwaukee', 'Wisconsin'),
              ('Cleveland', 'Ohio'), ('San Antonio', 'Texas'), ('Baltimore', 'Maryland'),
              ('Kansas City', 'Missouri'), ('Virginia Beach', 'Virginia'), ('Omaha', 'Nebraska')]:

        city_name, state_abbrev = e[0].lower().replace(" ", "_"), us_state_to_abbrev[e[1]].lower()
        city_format = e[0] + ' (' + us_state_to_abbrev[e[1]] + ')'
        if os.path.exists(file_path + city_name + '_' + state_abbrev + '/processed'):
            shutil.rmtree(file_path + city_name + '_' + state_abbrev + '/processed')
        dataset = LoadDataset(file_path, city_name + '_' + state_abbrev)
        data = dataset[0]
        class_num = dataset.num_classes

        # 60%, 20% and 20% for training, validation and test
        data.train_mask, data.val_mask, data.test_mask = train_test_split_stratify(dataset, train_ratio=0.6,
                                                                                   val_ratio=0.2,
                                                                                   class_num=class_num)
        sc = MinMaxScaler()
        data.x[data.train_mask] = torch.tensor(sc.fit_transform(data.x[data.train_mask]), dtype=torch.float)
        data.x[data.val_mask] = torch.tensor(sc.transform(data.x[data.val_mask]), dtype=torch.float)
        data.x[data.test_mask] = torch.tensor(sc.transform(data.x[data.test_mask]), dtype=torch.float)

        edge_attr_all = MinMaxScaler().fit_transform(data.edge_attr.cpu())
        edge_attr_all = torch.tensor(edge_attr_all).float().to(device)

        coords = data.coords.numpy()
        gdf_pred = pd.DataFrame({'x': coords[:, 0], 'y': coords[:, 1], 'label': data.y.numpy()})
        zip_iterator = zip(gdf_pred.index, gdf_pred[['x', 'y']].values)
        pos_dict = dict(zip_iterator)
        draw_with_labels(gdf_pred, 'Ground Truth', pos_dict=pos_dict, )

        X_train, X_test, y_train, y_test = data.x[data.train_mask].cpu().numpy(), data.x[data.test_mask].cpu().numpy(), \
            data.y[data.train_mask].cpu().numpy(), data.y[data.test_mask].cpu().numpy()

        data = data.to(device)

        start_time = time.time()
        model = MLP().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
        res = train_loop(model, data, optimizer, num_epochs, model_name='MLP', pos_dict=pos_dict, city_name=city_name)
        t = round(time.time() - start_time, 2)
        all_res.append((city_format,) + ('MLP',) + res + (t,))
        print("Execution time: %.4f seconds" % t)

        start_time = time.time()
        model = GCN().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
        res = train_loop(model, data, optimizer, num_epochs, model_name='GCN', pos_dict=pos_dict, city_name=city_name)
        t = round(time.time() - start_time, 2)
        all_res.append((city_format,) + ('GCN',) + res + (t,))
        print("Execution time: %.4f seconds" % t)

        start_time = time.time()
        model = ChebNet().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.02, weight_decay=5e-4)
        res = train_loop(model, data, optimizer, num_epochs, model_name='ChebNet', pos_dict=pos_dict, city_name=city_name)
        all_res.append((city_format,) + ('ChebNet',) + res + (t,))
        print("Execution time: %.4f seconds" % t)

        start_time = time.time()
        model = GraphSAGE().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
        res = train_loop(model, data, optimizer, num_epochs, model_name='GraphSAGE', pos_dict=pos_dict, city_name=city_name)
        t = round(time.time() - start_time, 2)
        all_res.append((city_format,) + ('GraphSAGE',) + res + (t,))
        print("Execution time: %.4f seconds" % t)

        start_time = time.time()
        model = GAT().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
        res = train_loop(model, data, optimizer, num_epochs, model_name='GAT', pos_dict=pos_dict,city_name=city_name)
        t = round(time.time() - start_time, 2)
        all_res.append((city_format,) + ('GAT',) + res + (t,))
        print("Execution time: %.4f seconds" % t)

        start_time = time.time()
        model = GraphTransformer().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
        res = train_loop(model, data, optimizer, num_epochs, model_name='Graphormer', pos_dict=pos_dict,city_name=city_name)
        t = round(time.time() - start_time, 2)
        all_res.append((city_format,) + ('Transformer',) + res + (t,))
        print("Execution time: %.4f seconds" % t)

        component_dir = np.concatenate((data.edge_attr.cpu(), data.edge_attr_dir.cpu()), axis=1)
        component_ang = np.concatenate((data.edge_attr.cpu(), data.edge_attr_ang.cpu()), axis=1)
        component_dir = StandardScaler().fit_transform(component_dir)
        component_ang = StandardScaler().fit_transform(component_ang)
        data.component_dir = torch.tensor(component_dir).float().to(device)
        data.component_ang = torch.tensor(component_ang).float().to(device)

        start_time = time.time()
        model = TPmodel().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.006, weight_decay=5e-4)
        res = train_loop(model, data, optimizer, num_epochs, model_name='TP', pos_dict=pos_dict, city_name=city_name)
        t = round(time.time() - start_time, 2)
        all_res.append((city_format,) + ('TP',) + res + (t,))
        print("Execution time: %.4f seconds" % t)

if __name__ == '__main__':
    main()