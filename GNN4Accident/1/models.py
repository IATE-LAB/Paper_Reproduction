from .utils import *
from .dataloader import *

dataset = None
data = None
edge_attr_all = None


class TPConv(MessagePassing):
    """
    参数:
        in_channels (int 或 tuple): 每个输入样本的大小，或者设置为 :obj:`-1` 以从前向方法中的第一个输入推导大小。
            一个元组对应源和目标维度的尺寸。
        out_channels (int): 每个输出样本的大小。
        nn (torch.nn.Module): 多个非线性变换层，将形状为 :obj:`[-1, num_node_features + num_edge_features]` 的特征数据映射到形状 :obj:`[-1, new_dimension]` 的数据，
            例如由 :class:`torch.nn.Sequential` 定义。
        aggr (string, 可选): 要使用的聚合方案 (:obj:`"add"`, :obj:`"mean"`, :obj:`"max"`)。
            (默认: :obj:`"add"`)
        root_weight (bool, 可选): 如果设置为 :obj:`False`，层将不会将变换后的根节点特征添加到输出中。
            (默认: :obj:`True`)
        bias (bool, 可选): 如果设置为 :obj:`False`，层将不会学习一个加性偏置。
            (默认: :obj:`True`)
        **kwargs (可选): 额外的参数，用于 :class:`torch_geometric.nn.conv.MessagePassing`。
    """

    def __init__(self, in_channels: Union[int, Tuple[int, int]],
                 out_channels: int, nn: Callable, aggr: str = 'add',
                 root_weight: bool = True, bias: bool = True, **kwargs):
        super(TPConv, self).__init__(aggr=aggr, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.nn = nn
        self.aggr = aggr

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        self.in_channels_l = in_channels[0]

        if root_weight:
            self.root = Parameter(torch.Tensor(in_channels[1], out_channels))
        else:
            self.register_parameter('root', None)

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    # xavier初始化
    def reset_parameters(self):
        reset(self.nn)
        if self.root is not None:
            torch.nn.init.xavier_uniform_(self.root)
        zeros(self.bias)

    # 前向传播
    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                edge_attr: OptTensor = None, size: Size = None) -> Tensor:
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=size)  # 消息传递机制

        x_r = x[1]
        if x_r is not None and self.root is not None:
            out += torch.matmul(x_r, self.root)

        if self.bias is not None:
            out += self.bias
        return out

    # 消息传递汇聚邻居节点信息
    def message(self, x_i: Tensor, x_j: Tensor, edge_attr: Tensor) -> Tensor:
        inputs = torch.cat([x_j, edge_attr], dim=1)
        return self.nn(inputs)

    def __repr__(self):
        return '{}({}, {}, aggr="{}", nn={})'.format(self.__class__.__name__,
                                                     self.in_channels,
                                                     self.out_channels,
                                                     self.aggr, self.nn)


class MLP(nn.Module):
    def __init__(self, hidden_dim=d):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(dataset.num_features, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, dataset.num_classes)

    def forward(self):
        x = F.relu(self.fc1(data.x))
        x = F.dropout(x, p=p, training=self.training)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)


class GCN(torch.nn.Module):
    def __init__(self, hidden_dim=d):
        super(GCN, self).__init__()
        self.conv1 = pyg_nn.GCNConv(dataset.num_features, hidden_dim)
        self.conv2 = pyg_nn.GCNConv(hidden_dim, hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, dataset.num_classes)

    def forward(self):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=p, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        x = self.fc1(x)
        return F.log_softmax(x, dim=1)


class ChebNet(torch.nn.Module):
    def __init__(self, hidden_dim=d):
        super(ChebNet, self).__init__()
        self.conv1 = pyg_nn.ChebConv(dataset.num_features, hidden_dim, K=2)
        self.conv2 = pyg_nn.ChebConv(hidden_dim, hidden_dim, K=2)
        self.fc1 = nn.Linear(hidden_dim, dataset.num_classes)

    def forward(self):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=p, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        x = self.fc1(x)
        return F.log_softmax(x, dim=1)


class ARMANet(torch.nn.Module):
    def __init__(self, hidden_dim=d):
        super(ARMANet, self).__init__()
        self.conv1 = pyg_nn.ARMAConv(dataset.num_features, hidden_dim)
        self.conv2 = pyg_nn.ARMAConv(hidden_dim, hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, dataset.num_classes)

    def forward(self):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=p, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        x = self.fc1(x)
        return F.log_softmax(x, dim=1)


class GraphSAGE(torch.nn.Module):
    def __init__(self, dim=d):
        super(GraphSAGE, self).__init__()
        self.conv1 = pyg_nn.SAGEConv(dataset.num_features, dim)
        self.conv2 = pyg_nn.SAGEConv(dim, dim * 2, normalize=True)
        self.fc1 = nn.Linear(dim * 2, dataset.num_classes)

    def forward(self):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=p, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        x = self.fc1(x)
        return F.log_softmax(x, dim=1)


class TAGCN(torch.nn.Module):
    def __init__(self, hidden_dim=d):
        super(TAGCN, self).__init__()
        self.conv1 = pyg_nn.TAGConv(dataset.num_features, hidden_dim)
        self.conv2 = pyg_nn.TAGConv(hidden_dim, hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, dataset.num_classes)

    def forward(self):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=p, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        x = self.fc1(x)
        return F.log_softmax(x, dim=1)


class GIN(torch.nn.Module):
    def __init__(self, dim=d):
        super(GIN, self).__init__()
        nn1 = nn.Sequential(nn.Linear(dataset.num_features, dim * 2), nn.ReLU(), nn.Linear(dim * 2, dim))
        nn2 = nn.Sequential(nn.Linear(dim, dim * 2), nn.ReLU(), nn.Linear(dim * 2, dim))
        self.conv1 = pyg_nn.GINConv(nn1)
        self.conv2 = pyg_nn.GINConv(nn2)
        self.fc1 = nn.Linear(dim, dataset.num_classes)

    def forward(self):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=p, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        x = self.fc1(x)
        return F.log_softmax(x, dim=1)


class GAT(torch.nn.Module):
    def __init__(self, dim=d):
        super(GAT, self).__init__()
        self.conv1 = pyg_nn.GATConv(dataset.num_features, dim, edge_dim=edge_attr_all.shape[1])
        self.conv2 = pyg_nn.GATConv(dim, dim, edge_dim=edge_attr_all.shape[1])
        self.fc1 = nn.Linear(dim, dataset.num_classes)

    def forward(self):
        x, edge_index, edge_attr = data.x, data.edge_index, edge_attr_all
        x = F.relu(self.conv1(x, edge_index, edge_attr))
        x = F.dropout(x, p=p, training=self.training)
        x = F.relu(self.conv2(x, edge_index, edge_attr))
        x = self.fc1(x)
        return F.log_softmax(x, dim=1)


class MPNN(torch.nn.Module):
    def __init__(self, dim=d):
        super(MPNN, self).__init__()
        nn1 = nn.Sequential(nn.Linear(edge_attr_all.shape[1], 16), nn.ReLU(), nn.Linear(16, dataset.num_features * dim))
        self.conv1 = pyg_nn.NNConv(dataset.num_features, dim, nn1)
        nn2 = nn.Sequential(nn.Linear(edge_attr_all.shape[1], 16), nn.ReLU(), nn.Linear(16, dim * dim))
        self.conv2 = pyg_nn.NNConv(dim, dim, nn2)
        self.fc1 = nn.Linear(dim, dataset.num_classes)

    def forward(self):
        x, edge_index, edge_attr = data.x, data.edge_index, edge_attr_all  # data.edge_attr
        x = F.relu(self.conv1(x, edge_index, edge_attr))
        x = F.dropout(x, p=p, training=self.training)
        x = F.relu(self.conv2(x, edge_index, edge_attr))
        x = self.fc1(x)
        return F.log_softmax(x, dim=1)


class CGC(torch.nn.Module):
    def __init__(self, dim=d):
        super(CGC, self).__init__()
        self.conv1 = pyg_nn.CGConv(dataset.num_features, edge_attr_all.size(-1))
        self.conv2 = pyg_nn.CGConv(dataset.num_features, edge_attr_all.size(-1))
        self.fc1 = nn.Linear(dataset.num_features, dataset.num_classes)

    def forward(self):
        x, edge_index, edge_attr = data.x, data.edge_index, edge_attr_all
        x = F.relu(self.conv1(x, edge_index, edge_attr))  #
        x = F.dropout(x, p=p, training=self.training)
        x = F.relu(self.conv2(x, edge_index, edge_attr))
        x = self.fc1(x)
        return F.log_softmax(x, dim=-1)


class GraphTransformer(torch.nn.Module):
    def __init__(self, dim=d):
        super(GraphTransformer, self).__init__()
        self.conv1 = pyg_nn.TransformerConv(dataset.num_features, dim, edge_dim=edge_attr_all.shape[1])
        self.conv2 = pyg_nn.TransformerConv(dim, dim, edge_dim=edge_attr_all.shape[1])
        self.fc1 = nn.Linear(dim, dataset.num_classes)

    def forward(self):
        x, edge_index, edge_attr = data.x, data.edge_index, edge_attr_all
        x = F.relu(self.conv1(x, edge_index, edge_attr))
        x = F.dropout(x, p=p, training=self.training)
        x = F.relu(self.conv2(x, edge_index, edge_attr))
        x = self.fc1(x)
        return F.log_softmax(x, dim=1)


class GEN(torch.nn.Module):
    def __init__(self, dim=d):
        super(GEN, self).__init__()
        self.node_encoder = nn.Linear(data.x.size(-1), dim)
        self.edge_encoder = nn.Linear(edge_attr_all.size(-1), dim)
        self.conv1 = pyg_nn.GENConv(dim, dim)
        self.conv2 = pyg_nn.GENConv(dim, dim)
        self.fc1 = nn.Linear(dim, dataset.num_classes)

    def forward(self):
        x, edge_index, edge_attr = self.node_encoder(data.x), data.edge_index, self.edge_encoder(edge_attr_all)
        x = F.relu(self.conv1(x, edge_index, edge_attr))
        x = F.dropout(x, p=p, training=self.training)
        x = F.relu(self.conv2(x, edge_index, edge_attr))
        x = self.fc1(x)
        return F.log_softmax(x, dim=1)


class TPmodel(torch.nn.Module):
    def __init__(self, dim=d):
        super(TPmodel, self).__init__()
        convdim = 8
        self.node_encoder = nn.Sequential(nn.Linear(data.x.size(-1), dim), nn.LeakyReLU(), nn.Linear(dim, dim))
        self.edge_encoder_dir = nn.Sequential(nn.Linear(data.component_dir.size(-1), dim), nn.LeakyReLU(),
                                              nn.Linear(dim, dim))
        self.edge_encoder_ang = nn.Sequential(nn.Linear(data.component_ang.size(-1), dim), nn.LeakyReLU(),
                                              nn.Linear(dim, dim))
        nn1 = nn.Sequential(nn.Linear(dim + dim, dim), nn.LeakyReLU(), nn.Linear(dim, dim), nn.LeakyReLU(),
                            nn.Linear(dim, convdim))
        self.conv1 = TPConv(dim, convdim, nn1)
        nn2 = nn.Sequential(nn.Linear(2 * convdim + dim, dim), nn.LeakyReLU(), nn.Linear(dim, dim), nn.LeakyReLU(),
                            nn.Linear(dim, dataset.num_classes))
        self.conv2 = TPConv(2 * convdim, dataset.num_classes, nn2)
        self.bn1 = nn.BatchNorm1d(convdim * 2)
        nn1_2 = nn.Sequential(nn.Linear(dim + dim, dim), nn.LeakyReLU(), nn.Linear(dim, dim), nn.LeakyReLU(),
                              nn.Linear(dim, convdim))
        self.conv1_2 = TPConv(dim, convdim, nn1_2)
        nn2_2 = nn.Sequential(nn.Linear(2 * convdim + dim, dim), nn.LeakyReLU(), nn.Linear(dim, dim), nn.LeakyReLU(),
                              nn.Linear(dim, dataset.num_classes))
        self.conv2_2 = TPConv(2 * convdim, dataset.num_classes, nn2_2)
        self.bn2 = nn.BatchNorm1d(dataset.num_classes * 2)
        self.fc = nn.Linear(dataset.num_classes * 2, dataset.num_classes)

    def forward(self):
        x, edge_index = self.node_encoder(data.x), data.edge_index
        edge_attr_dir, edge_attr_ang = self.edge_encoder_dir(data.component_dir), self.edge_encoder_ang(
            data.component_ang)
        x1 = self.conv1(x, edge_index, edge_attr_dir)
        x1 = F.dropout(x1, p=p / 2, training=self.training)
        x1 = F.leaky_relu(x1, negative_slope=0.01)
        x2 = self.conv1_2(x, edge_index, edge_attr_ang)
        x2 = F.dropout(x2, p=p / 2, training=self.training)
        x2 = F.leaky_relu(x2, negative_slope=0.01)
        x = torch.cat((x1, x2), axis=1)
        x = self.bn1(x)
        x = F.dropout(x, p=p, training=self.training)
        x1 = F.leaky_relu(self.conv2(x, edge_index, edge_attr_dir), negative_slope=0.01)
        x2 = F.leaky_relu(self.conv2_2(x, edge_index, edge_attr_ang), negative_slope=0.01)
        x = torch.cat((x1, x2), axis=1)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)