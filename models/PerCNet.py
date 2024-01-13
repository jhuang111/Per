
import math
from typing import Optional
from .features import angle_emb, torsion_emb
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pydantic.typing import Literal
from torch_geometric.nn import Linear, MessagePassing, global_mean_pool
from torch_geometric.nn.models.schnet import ShiftedSoftplus

from models.base import BaseSettings
from models.transformer import TransformerConv

from models.utils import RBFExpansion

device = torch.device("cuda:2")

from torch_cluster import radius_graph
from torch_geometric.nn import GraphConv, GraphNorm
from torch_geometric.nn import inits

from .features import angle_emb, torsion_emb

from torch_scatter import scatter, scatter_min

from torch.nn import Embedding

import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F

import math
from math import sqrt

try:
    import sympy as sym
except ImportError:
    sym = None


def swish(x):
    return x * torch.sigmoid(x)


class Linear(torch.nn.Module):

    def __init__(self, in_channels, out_channels, bias=True,
                 weight_initializer='glorot',
                 bias_initializer='zeros'):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight_initializer = weight_initializer
        self.bias_initializer = bias_initializer

        assert in_channels > 0
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels))

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        if self.in_channels > 0:
            if self.weight_initializer == 'glorot':
                inits.glorot(self.weight)
            elif self.weight_initializer == 'glorot_orthogonal':
                inits.glorot_orthogonal(self.weight, scale=2.0)
            elif self.weight_initializer == 'uniform':
                bound = 1.0 / math.sqrt(self.weight.size(-1))
                torch.nn.init.uniform_(self.weight.data, -bound, bound)
            elif self.weight_initializer == 'kaiming_uniform':
                inits.kaiming_uniform(self.weight, fan=self.in_channels,
                                      a=math.sqrt(5))
            elif self.weight_initializer == 'zeros':
                inits.zeros(self.weight)
            elif self.weight_initializer is None:
                inits.kaiming_uniform(self.weight, fan=self.in_channels,
                                      a=math.sqrt(5))
            else:
                raise RuntimeError(
                    f"Linear layer weight initializer "
                    f"'{self.weight_initializer}' is not supported")

        if self.in_channels > 0 and self.bias is not None:
            if self.bias_initializer == 'zeros':
                inits.zeros(self.bias)
            elif self.bias_initializer is None:
                inits.uniform(self.in_channels, self.bias)
            else:
                raise RuntimeError(
                    f"Linear layer bias initializer "
                    f"'{self.bias_initializer}' is not supported")

    def forward(self, x):
        """"""
        # print(x.dtype)
        # print(self.weight.dtype)
        # print(self.bias.dtype)
        x = x.to(torch.float32)
        return F.linear(x, self.weight, self.bias)


class TwoLayerLinear(torch.nn.Module):
    def __init__(
            self,
            in_channels,
            middle_channels,
            out_channels,
            bias=False,
            act=False,
    ):
        super(TwoLayerLinear, self).__init__()
        self.lin1 = Linear(in_channels, middle_channels, bias=bias)
        self.lin2 = Linear(middle_channels, out_channels, bias=bias)
        self.act = act

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, x):
        x = self.lin1(x)
        if self.act:
            x = swish(x)
        x = self.lin2(x)
        if self.act:
            x = swish(x)
        return x


class EmbeddingBlock(torch.nn.Module):
    def __init__(self, hidden_channels, act=swish):
        super(EmbeddingBlock, self).__init__()
        self.act = act
        self.emb = Embedding(95, hidden_channels)
        self.reset_parameters()

    def reset_parameters(self):
        self.emb.weight.data.uniform_(-sqrt(3), sqrt(3))

    def forward(self, x):
        x = self.act(self.emb(x))
        return x


class EdgeGraphConv(GraphConv):

    def message(self, x_j, edge_weight) -> Tensor:
        return x_j if edge_weight is None else edge_weight * x_j


class PerCNetConfig(BaseSettings):
    name: Literal["PerCNet"]
    conv_layers: int = 3
    atom_input_features: int = 92
    inf_edge_features: int = 64
    fc_features: int = 256
    output_dim: int = 256
    output_features: int = 1
    rbf_min = -4.0
    rbf_max = 4.0
    potentials = []
    charge_map = False
    transformer = False

    class Config:
        """Configure model settings behavior."""
        env_prefix = "jv_model"


class PerCNetConv(MessagePassing):

    def __init__(self, fc_features,
                 num_layers=4,
                 hidden_channels=256,
                 middle_channels=64,
                 out_channels=1,
                 num_radial=3,
                 num_spherical=2,
                 num_output_layers=3,
                 output_channels=256,
                 act=swish):
        super(PerCNetConv, self).__init__(node_dim=0)
        self.bn = nn.BatchNorm1d(fc_features)
        self.bn_interaction = nn.BatchNorm1d(fc_features)
        self.nonlinear_full = nn.Sequential(
            nn.Linear(3 * fc_features, fc_features),
            nn.SiLU(),
            nn.Linear(fc_features, fc_features)
        )
        self.nonlinear = nn.Sequential(
            nn.Linear(3 * fc_features, fc_features),
            nn.SiLU(),
            nn.Linear(fc_features, fc_features),
        )
        self.act = act

        self.conv1 = EdgeGraphConv(hidden_channels, hidden_channels)

        self.conv2 = EdgeGraphConv(hidden_channels, hidden_channels)

        self.lin1 = Linear(hidden_channels, hidden_channels)

        self.lin2 = Linear(hidden_channels, hidden_channels)

        self.lin_cat = Linear(2 * hidden_channels, hidden_channels)

        self.norm = GraphNorm(hidden_channels)

        # Transformations of Bessel and spherical basis representations.
        self.lin_feature1 = TwoLayerLinear(num_radial * num_spherical ** 2, middle_channels, hidden_channels)
        self.lin_feature2 = TwoLayerLinear(num_radial * num_spherical, middle_channels, hidden_channels)

        # Dense transformations of input messages.
        self.lin = Linear(hidden_channels, hidden_channels)
        self.lins = torch.nn.ModuleList()
        for _ in range(num_layers):
            self.lins.append(Linear(hidden_channels, hidden_channels))
        self.final = Linear(hidden_channels, output_channels)

    def period_come(self, x, feature1, feature2, edge_index, batch):
        x = self.act(self.lin(x))
        feature1 = self.lin_feature1(feature1)
        h1 = self.conv1(x, edge_index, feature1)
        h1 = self.lin1(h1)
        h1 = self.act(h1)

        feature2 = self.lin_feature2(feature2)
        h2 = self.conv2(x, edge_index, feature2)
        h2 = self.lin2(h2)
        h2 = self.act(h2)

        h = self.lin_cat(torch.cat([h1, h2], 1))

        h = h + x
        for lin in self.lins:
            h = self.act(lin(h)) + h
        h = self.norm(h, batch)
        h = self.final(h)
        return h

    def forward(self, x, edge_index, edge_attr):
        # result_p = self.period_come(x, feature1=feature1, feature2=feature2, edge_index=period_edge_index)

        out = self.propagate(
            edge_index, x=x, edge_attr=edge_attr, size=(x.size(0), x.size(0))
        )
        # result_o = self.bn(out)
        # print('out:')
        # print(out)
        # TODO: ComeENet与PerCNet的结果如何进行处理？
        return torch.relu(x + self.bn(out))
        # return torch.relu(x + result_o)

    def message(self, x_i, x_j, edge_attr, index):
        edge_attr = edge_attr.to(torch.float32)
        # print(x_i.size())
        # print(x_j.size())
        # print(edge_attr.size())
        score = torch.sigmoid(self.bn_interaction(self.nonlinear_full(torch.cat((x_i, x_j, edge_attr), dim=1))))
        return score * self.nonlinear(torch.cat((x_i, x_j, edge_attr), dim=1))


hidden_channels = 256
output_channels = 256


class PerCNet(nn.Module):

    def __init__(self, config: PerCNetConfig = PerCNetConfig(name="PerCNet")):
        super().__init__()
        self.config = config
        if not config.charge_map:
            self.atom_embedding = nn.Linear(
                config.atom_input_features, config.fc_features
            )
        else:
            self.atom_embedding = nn.Linear(
                config.atom_input_features + 10, config.fc_features
            )

        # self.infinite = True

        self.edge_embedding = nn.Sequential(
            RBFExpansion(
                vmin=config.rbf_min,
                vmax=config.rbf_max,
                bins=config.fc_features,
            ),
            nn.Linear(config.fc_features, config.fc_features),
            nn.SiLU(),
        )

        self.inf_edge_embedding = RBFExpansion(
            vmin=config.rbf_min,
            vmax=config.rbf_max,
            bins=config.inf_edge_features,
            type='multiquadric'
        )

        self.period_embedding = nn.Sequential(nn.SiLU(), )
        self.lin = Linear(256, 256)
        self.infinite_linear = nn.Linear(config.inf_edge_features, config.fc_features)

        self.infinite_bn = nn.BatchNorm1d(config.fc_features)

        self.feature1 = torsion_emb(num_radial=3, num_spherical=2, cutoff=4.0)
        self.conv1 = EdgeGraphConv(hidden_channels, hidden_channels)

        self.conv2 = EdgeGraphConv(hidden_channels, hidden_channels)

        self.lin1 = Linear(hidden_channels, hidden_channels)

        self.lin2 = Linear(hidden_channels, hidden_channels)
        self.norm = GraphNorm(hidden_channels)
        self.final = Linear(hidden_channels, output_channels)
        self.lin_cat = Linear(2 * hidden_channels, hidden_channels)
        self.feature2 = angle_emb(num_radial=3, num_spherical=2, cutoff=4.0)
        # n**2 = 4 n**2*k=12
        self.lin_feature1 = TwoLayerLinear(12, 64, 256)
        # n=2 n*k=6
        self.lin_feature2 = TwoLayerLinear(6, 64, 256)
        self.conv_layers = nn.ModuleList(
            [
                PerCNetConv(config.fc_features)
                for _ in range(config.conv_layers)
            ]
        )
        if config.transformer:
            self.transformer_conv_layers = nn.ModuleList(
                [
                    TransformerConv(config.fc_features, config.fc_features)
                    for _ in range(config.conv_layers)
                ]
            )

        self.fc = nn.Sequential(
            nn.Linear(config.fc_features, config.fc_features), ShiftedSoftplus()
        )
        self.lins = torch.nn.ModuleList()
        for _ in range(3):
            self.lins.append(Linear(256, 256))
        self.lin_out = Linear(256, 1)
        self.act = swish
        self.fc_out = nn.Linear(config.output_dim, config.output_features)

    # TODO:
    #  ②修改网络输入：PerCNet的输入需要加入TBF与SBF
    #  ③修改网络：1.需要全局卷积和局部卷积对tbf和sbf进行处理；2.每个交互层结束后需要将周期节点进行节点特征一致性处理
    def forward(self, data, print_data=False):
        # 周期同化，这里假定每两个节点都是周期节点，直接取平均
        def period_come1(x, feature1, feature2, edge_index, batch):
            x = self.act(self.lin(x))
            feature1 = self.lin_feature1(feature1)
            h1 = self.conv1(x, edge_index, feature1)
            h1 = self.lin1(h1)
            h1 = self.act(h1)

            feature2 = self.lin_feature2(feature2)
            h2 = self.conv2(x, edge_index, feature2)
            h2 = self.lin2(h2)
            h2 = self.act(h2)

            h = self.lin_cat(torch.cat([h1, h2], 1))

            h = h + x
            for lin in self.lins:
                h = self.act(lin(h)) + h
            h = self.norm(h, batch)
            h = self.final(h)
            return h

        def period_come2(x, feature1, feature2, edge_index, batch):
            x = self.act(self.lin(x))
            feature1 = self.lin_feature1(feature1)
            h1 = self.conv1(x, edge_index, feature1)
            h1 = self.lin1(h1)
            h1 = self.act(h1)

            feature2 = self.lin_feature2(feature2)
            h2 = self.conv2(x, edge_index, feature2)
            h2 = self.lin2(h2)
            h2 = self.act(h2)

            h = self.lin_cat(torch.cat([h1, h2], 1))

            h = h + x
            for lin in self.lins:
                h = self.act(lin(h)) + h
            h = self.norm(h, batch)
            h = self.final(h)
            return h

        def period_come3(x, feature1, feature2, edge_index, batch):
            x = self.act(self.lin(x))
            feature1 = self.lin_feature1(feature1)
            h1 = self.conv1(x, edge_index, feature1)
            h1 = self.lin1(h1)
            h1 = self.act(h1)

            feature2 = self.lin_feature2(feature2)
            h2 = self.conv2(x, edge_index, feature2)
            h2 = self.lin2(h2)
            h2 = self.act(h2)

            h = self.lin_cat(torch.cat([h1, h2], 1))

            h = h + x
            for lin in self.lins:
                h = self.act(lin(h)) + h
            h = self.norm(h, batch)
            h = self.final(h)
            return h

        def period_come4(x, feature1, feature2, edge_index, batch):
            x = self.act(self.lin(x))
            feature1 = self.lin_feature1(feature1)
            h1 = self.conv1(x, edge_index, feature1)
            h1 = self.lin1(h1)
            h1 = self.act(h1)

            feature2 = self.lin_feature2(feature2)
            h2 = self.conv2(x, edge_index, feature2)
            h2 = self.lin2(h2)
            h2 = self.act(h2)

            h = self.lin_cat(torch.cat([h1, h2], 1))

            h = h + x
            for lin in self.lins:
                h = self.act(lin(h)) + h
            h = self.norm(h, batch)
            h = self.final(h)
            return h

        def assi01(x):
            updated_tmp = ((x[::2] + x[1::2]) / 2.0).to(torch.double).to(device)
            updated_tmp = updated_tmp.to(device)
            x[0::2] = updated_tmp.clone()
            x[1::2] = updated_tmp.clone()
            # x = x.to(torch.float32)
            return x

        """CGCNN function mapping graph to outputs."""

        cal_inf = True
        cal_edge = True
        cal_4edge = True
        use_ComENet = False
        if cal_edge:
            edge_index = data.edge_index.to(torch.int64)
            # 固定边特征计算，使用RBF扩展的键长
            edge_attr = data.edge_attr.to(torch.float32)
            edge_features = self.edge_embedding(-0.75 / edge_attr)
            # print('edge features:')
            # print(edge_features)
            # print(edge_features.size())
        # print('edge features')
        # print(edge_features)
        # TODO:前面周期扩展会不会对该步造成影响？
        if cal_inf:
            # 无限边索引
            inf_edge_index = data.inf_edge_index
            # 计算无限边特征 potentials: [-0.801, -0.074, 0.145]。potentials有什么作用？？？
            inf_feat = sum([data.inf_edge_attr[:, i] * pot for i, pot in enumerate(self.config.potentials)])
            # 对无限边特征做embedding和bn
            inf_edge_features = self.inf_edge_embedding(inf_feat)
            inf_edge_features = self.infinite_bn(F.softplus(self.infinite_linear(inf_edge_features)))
            # print('inf edge features')
            # print(inf_edge_features)

        # initial node features: atom feature network...
        if self.config.charge_map:  # 原子特征与组信息拼接
            node_features = self.atom_embedding(torch.cat([data.x, data.g_feats], -1))
        else:  # 默认false
            node_features = self.atom_embedding(data.x)
        if cal_4edge:
            # print(data.size())  # (1072, 1072)
            dist = data.dist

            theta = data.theta
            phi = data.phi
            TBF = self.feature1(dist, theta, phi)
            feature1 = self.lin_feature1(TBF)

            #tau = data.tau
            #SBF = self.feature2(dist, tau)
            #feature2 = self.lin_feature2(SBF)

            tuple_index = data.tuple_edge_index

        
        if self.config.transformer:  # 默认False
            edge_index = torch.cat([edge_index, inf_edge_index], 1)  # 拼接边索引
            edge_features = torch.cat([edge_features, inf_edge_features], 0)  # 拼接边特征
            tuple_edge_index = torch.cat([tuple_index, tuple_index], 1)  # 拼接边索引
            tuple_edge_features = torch.cat([feature1, feature2], 0)  # 拼接边特征
        else:    
            # edge_index = torch.cat([edge_index, inf_edge_index, tuple_index, tuple_index], 1)  # 拼接边索引
            # edge_features = torch.cat([edge_features, inf_edge_features, feature1, feature2], 0)  # 拼接边特征
            # edge_index = torch.cat([edge_index, inf_edge_index], 1)  # 拼接边索引
            # edge_features = torch.cat([edge_features, inf_edge_features], 0)  # 拼接边特征
            # edge_index = torch.cat([edge_index, tuple_index], 1)  # 拼接边索引
            # edge_features = torch.cat([edge_features, feature1], 0)  # 拼接边特征
            pass
        if use_ComENet:
            node_features = period_come1(x=node_features, feature1=TBF, feature2=SBF, edge_index=tuple_index,
                                         batch=data.batch)
            node_features = period_come2(x=node_features, feature1=TBF, feature2=SBF, edge_index=tuple_index,
                                         batch=data.batch)
            node_features = period_come3(x=node_features, feature1=TBF, feature2=SBF, edge_index=tuple_index,
                                         batch=data.batch)
            node_features = period_come4(x=node_features, feature1=TBF, feature2=SBF, edge_index=tuple_index,
                                         batch=data.batch)
            node_features = assi01(node_features)
            for lin in self.lins:
                node_features = self.act(lin(node_features))
            node_features = self.lin_out(node_features)

            result = scatter(node_features, data.batch, dim=0)
            return result
        # PerCNet的处理方式：
        for i in range(self.config.conv_layers):
            if self.config.transformer:  # 默认False
                local_node_features = self.conv_layers[i](node_features, edge_index, edge_features)
                inf_node_features = self.transformer_conv_layers[i](node_features, tuple_edge_index, tuple_edge_features)
                node_features = local_node_features + inf_node_features
                # node_features = self.transformer_conv_layers[i](node_features, tuple_edge_index, tuple_edge_features)
            else:  # 默认执行，对节点特征进行卷积
                # TODO: Ⅲ. ①对下面函数进行修改，添加参数TBF和SBF，网络中需要添加相应的卷积层
                node_features = self.conv_layers[i](node_features, edge_index, edge_features)
                # node_features = self.conv_layers[i](node_features, tuple_index, feature1)
                # node_features_clone = node_features.clone()-
                # node_features = assi01(node_features_clone)
                # TODO: Ⅲ. ②交互模块每次结束对源节点和周期节点的node_features做一致性处理
        # 全局池化
        features = global_mean_pool(node_features, data.batch)
        # 接一个全连接层
        features = self.fc(features)
        result = torch.squeeze(self.fc_out(features))

        return result
