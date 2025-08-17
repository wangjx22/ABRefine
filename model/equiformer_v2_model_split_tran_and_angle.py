import logging
import time
import math
import numpy as np
import torch
import torch.nn as nn
from pyexpat.model import XML_CQUANT_OPT
import time
import torch.nn.functional as F

from ocpmodels.common.utils import conditional_grad
from ocpmodels.models.base import BaseModel
from ocpmodels.models.scn.sampling import CalcSpherePoints
from ocpmodels.models.scn.smearing import (
    GaussianSmearing,
    LinearSigmoidSmearing,
    SigmoidSmearing,
    SiLUSmearing,
)

try:
    from e3nn import o3
except ImportError:
    pass

from model.equiformerv2.gaussian_rbf import GaussianRadialBasisLayer
from torch.nn import Linear
from model.equiformerv2.edge_rot_mat import init_edge_rot_mat
from model.equiformerv2.so3 import (
    CoefficientMappingModule,
    SO3_Embedding,
    SO3_Grid,
    SO3_Rotation,
    SO3_LinearV2
)

from model.equiformerv2.module_list import ModuleListInfo
from model.equiformerv2.so2_ops import SO2_Convolution
from model.equiformerv2.radial_function import RadialFunction
from model.equiformerv2.layer_norm import (
    EquivariantLayerNormArray,
    EquivariantLayerNormArraySphericalHarmonics,
    EquivariantRMSNormArraySphericalHarmonics,
    EquivariantRMSNormArraySphericalHarmonicsV2,
    get_normalization_layer
)

from model.equiformerv2.transformer_block import (
    SO2EquivariantGraphAttention,
    FeedForwardNetwork,
    TransBlockV2,
)
from model.equiformerv2.input_block import EdgeDegreeEmbedding

# Statistics of IS2RE 100K
_AVG_NUM_NODES = 77.81317
_AVG_DEGREE = 23.395238876342773  # IS2RE: 100k, max_radius = 5, max_neighbors = 100


class EquivariantCrossAttention(nn.Module):
    def __init__(self, feature_dim, num_heads=1, dropout=0.1):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.head_dim = feature_dim // num_heads
        self.scale = float(self.head_dim ** -0.5)
        
        # 简化注意力机制，移除投影层
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, cdr3_features, all_atom_features):
        batch_size = cdr3_features.size(0)
        
        # 计算注意力权重
        Q = cdr3_features
        K = all_atom_features
        V = all_atom_features
        
        # 计算相对特征
        Q_mean = Q.mean(dim=1, keepdim=True)
        K_mean = K.mean(dim=1, keepdim=True)
        
        Q_relative = Q - Q_mean
        K_relative = K - K_mean
        V_relative = V - K_mean
        
        # 计算注意力分数
        attn_scores = torch.matmul(Q_relative, K_relative.transpose(-2, -1)) * self.scale
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)
        
        # 注意力加权求和
        context = torch.matmul(attn_probs, V_relative)
        
        # 恢复平移
        output = context + Q_mean
        
        return output

class EquivariantFFN(nn.Module):
    def __init__(self, feature_dim, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        return x  # 简单地返回输入，保持等变性

class EquivariantTransformerLayer(nn.Module):
    def __init__(self, feature_dim, num_heads, dropout):
        super().__init__()
        self.attention = EquivariantCrossAttention(feature_dim, num_heads, dropout)
        self.ffn = EquivariantFFN(feature_dim, dropout=dropout)

    def forward(self, x, all_atom_features):
        # 仅使用注意力层
        x = self.attention(x, all_atom_features)
        return x

class EquivariantCDR3Processor(nn.Module):
    def __init__(self, feature_dim, num_heads=1, num_layers=4, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            EquivariantTransformerLayer(feature_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
    def forward(self, cdr3_features, all_atom_features):
        """ Args:
            cdr3_features: [batch_size, num_cdr3_atoms, feature_dim]
            all_atom_features: [batch_size, num_all_atoms, feature_dim]
        Returns:
            Updated CDR3 features: [batch_size, num_cdr3_atoms, feature_dim]
        """
        x = cdr3_features
        for layer in self.layers:
            x = layer(x, all_atom_features)
        return x

class InvariantCrossAttention(nn.Module):
    def __init__(self, feature_dim, num_heads=1, dropout=0.1):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.head_dim = feature_dim // num_heads
        
        assert self.head_dim * num_heads == feature_dim, "feature_dim must be divisible by num_heads"
        
        # 无偏置投影保持等变性（后续池化实现不变性）
        self.q_proj = nn.Linear(feature_dim, feature_dim, bias=False)
        self.k_proj = nn.Linear(feature_dim, feature_dim, bias=False)
        self.v_proj = nn.Linear(feature_dim, feature_dim, bias=False)
        
        # 不变性池化层
        self.pool = nn.AdaptiveAvgPool1d(1)  # 全局平均池化
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(feature_dim)
        
    def forward(self, cdr3_features, all_atom_features):
        """
        Args:
            cdr3_features: [batch_size, num_cdr3_atoms, feature_dim]
            all_atom_features: [batch_size, num_all_atoms, feature_dim]
        Returns:
            Invariant CDR3 features: [batch_size, num_cdr3_atoms, feature_dim]
        """
        batch_size = cdr3_features.size(0)
        
        # 投影得到Q、K、V
        Q = self.q_proj(cdr3_features)  # [B, N_cdr3, D]
        K = self.k_proj(all_atom_features)  # [B, N_all, D]
        V = self.v_proj(all_atom_features)  # [B, N_all, D]
        
        # 多头分割
        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 计算注意力分数
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)
        
        # 注意力加权求和
        context = torch.matmul(attn_probs, V)
        context = context.transpose(1, 2).contiguous()
        context = context.view(batch_size, -1, self.feature_dim)
        
        # 添加全局不变性：沿原子维度池化
        pooled_context = self.pool(context.transpose(1, 2)).transpose(1, 2)  # [B, 1, D]
        pooled_context = pooled_context.expand_as(cdr3_features)  # 广播到原子维度
        
        # 残差连接和LayerNorm
        output = self.layer_norm(pooled_context + cdr3_features)
        return output


class InvariantTransformerLayer(nn.Module):
    def __init__(self, feature_dim, num_heads, dropout):
        super().__init__()
        # 注意力子层
        self.attention = InvariantCrossAttention(feature_dim, num_heads, dropout)
        self.attn_norm = nn.LayerNorm(feature_dim)
        
        # 前馈子层
        self.ffn = nn.Sequential(
            nn.Linear(feature_dim, feature_dim*10, bias=False),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim*10, feature_dim, bias=False)
        )
        self.ffn_norm = nn.LayerNorm(feature_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, all_atom_features):
        # 注意力子层（带残差）
        attn_out = self.attention(x, all_atom_features)
        x = self.attn_norm(x + self.dropout(attn_out))
        
        # 前馈子层（带残差）
        ffn_out = self.ffn(x)
        x = self.ffn_norm(x + self.dropout(ffn_out))
        return x

class InvariantCDR3Processor(nn.Module):
    def __init__(self, feature_dim, num_heads=1, num_layers=4, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            InvariantTransformerLayer(feature_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        
    def forward(self, cdr3_features, all_atom_features):
        x = cdr3_features
        for layer in self.layers:
            x = layer(x, all_atom_features)
        return x

class InvariantCrossAttention(nn.Module):
    def __init__(self, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, cdr3_features, all_atom_features):
        """
        Args:
            cdr3_features: [batch_size, num_cdr3_atoms, 1]
            all_atom_features: [batch_size, num_all_atoms, 1]
        Returns:
            Invariant features: [batch_size, 1]
        """
        # 计算相对特征
        cdr3_mean = cdr3_features.mean(dim=1, keepdim=True)
        all_atom_mean = all_atom_features.mean(dim=1, keepdim=True)
        
        Q = cdr3_features - cdr3_mean
        K = all_atom_features - all_atom_mean
        
        # 使用平方距离计算注意力分数，确保反射不变性
        Q_squared = Q.pow(2)  # [batch_size, num_cdr3_atoms, 1]
        K_squared = K.pow(2)  # [batch_size, num_all_atoms, 1]
        
        # 计算配对距离
        distances = Q_squared + K_squared.transpose(-2, -1)  # [batch_size, num_cdr3_atoms, num_all_atoms]
        
        # 使用softmax获取注意力权重
        attn_weights = F.softmax(-distances, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 计算加权平均
        V = K_squared  # 使用平方特征作为值
        context = torch.matmul(attn_weights, V)  # [batch_size, num_cdr3_atoms, 1]
        
        # 全局平均池化
        output = context.mean(dim=1)  # [batch_size, 1]
        
        return output

class InvariantCDR3Processor(nn.Module):
    def __init__(self, num_layers=4, dropout=0.1):
        super().__init__()
        # self.attention = InvariantCrossAttention(dropout)
        self.layers = nn.ModuleList([
            InvariantCrossAttention(dropout)
            for _ in range(num_layers)
        ])
        
    def forward(self, cdr3_features, all_atom_features):
        """
        Args:
            cdr3_features: [batch_size, num_cdr3_atoms, 1]
            all_atom_features: [batch_size, num_all_atoms, 1]
        Returns:
            Invariant output: [batch_size, 1]
        """
        x = cdr3_features
        for layer in self.layers:
            x = layer(x, all_atom_features)
        return x
        # return self.attention(cdr3_features, all_atom_features)



class EquiformerV2(BaseModel):
    """
    Equiformer with graph attention built upon SO(2) convolution and feedforward network built upon S2 activation

    Args:
        use_pbc (bool):         Use periodic boundary conditions
        regress_forces (bool):  Compute forces
        otf_graph (bool):       Compute graph On The Fly (OTF)
        max_neighbors (int):    Maximum number of neighbors per atom
        max_radius (float):     Maximum distance between nieghboring atoms in Angstroms
        max_num_elements (int): Maximum atomic number

        num_layers (int):             Number of layers in the GNN
        sphere_channels (int):        Number of spherical channels (one set per resolution)
        attn_hidden_channels (int): Number of hidden channels used during SO(2) graph attention
        num_heads (int):            Number of attention heads
        attn_alpha_head (int):      Number of channels for alpha vector in each attention head
        attn_value_head (int):      Number of channels for value vector in each attention head
        ffn_hidden_channels (int):  Number of hidden channels used during feedforward network
        norm_type (str):            Type of normalization layer (['layer_norm', 'layer_norm_sh', 'rms_norm_sh'])

        lmax_list (int):              List of maximum degree of the spherical harmonics (1 to 10)
        mmax_list (int):              List of maximum order of the spherical harmonics (0 to lmax)
        grid_resolution (int):        Resolution of SO3_Grid

        num_sphere_samples (int):     Number of samples used to approximate the integration of the sphere in the output blocks

        edge_channels (int):                Number of channels for the edge invariant features
        use_atom_edge_embedding (bool):     Whether to use atomic embedding along with relative distance for edge scalar features
        share_atom_edge_embedding (bool):   Whether to share `atom_edge_embedding` across all blocks
        use_m_share_rad (bool):             Whether all m components within a type-L vector of one channel share radial function weights
        distance_function ("gaussian", "sigmoid", "linearsigmoid", "silu"):  Basis function used for distances

        attn_activation (str):      Type of activation function for SO(2) graph attention
        use_s2_act_attn (bool):     Whether to use attention after S2 activation. Otherwise, use the same attention as Equiformer
        use_attn_renorm (bool):     Whether to re-normalize attention weights
        ffn_activation (str):       Type of activation function for feedforward network
        use_gate_act (bool):        If `True`, use gate activation. Otherwise, use S2 activation
        use_grid_mlp (bool):        If `True`, use projecting to grids and performing MLPs for FFNs.
        use_sep_s2_act (bool):      If `True`, use separable S2 activation when `use_gate_act` is False.

        alpha_drop (float):         Dropout rate for attention weights
        drop_path_rate (float):     Drop path rate
        proj_drop (float):          Dropout rate for outputs of attention and FFN in Transformer blocks

        weight_init (str):          ['normal', 'uniform'] initialization of weights of linear layers except those in radial functions
    """

    def __init__(
            self,
            use_pbc=False,
            regress_forces=True,
            use_equivariant_block=True, 
            use_CDR3_att_block=False,
            use_invariant_block=False,
            use_invariant_block_for_tran=False,
            use_deep_net=False,
            use_KL_loss=False,
            otf_graph=True,
            max_neighbors=40,
            max_radius=12.0,
            max_num_elements=90,
            max_num_atom_names=37,
            max_num_residues=21,

            num_layers=2,
            sphere_channels=24,
            attn_hidden_channels=64,
            num_heads=2,
            attn_alpha_channels=32,
            attn_value_channels=8,
            ffn_hidden_channels=64,

            norm_type='layer_norm_sh',

            lmax_list=[2],
            mmax_list=[2],
            grid_resolution=14,

            num_sphere_samples=32,

            edge_channels=32,
            use_atom_edge_embedding=True,
            share_atom_edge_embedding=False,
            use_m_share_rad=False,
            distance_function="gaussian",
            num_distance_basis=512,

            attn_activation='silu',
            use_s2_act_attn=False,
            use_attn_renorm=True,
            ffn_activation='silu',
            use_gate_act=False,
            use_grid_mlp=True,
            use_sep_s2_act=True,

            alpha_drop=0.1,
            drop_path_rate=0.05,
            proj_drop=0.0,

            weight_init='uniform',
            **kwargs
    ):
        super().__init__()

        self.use_pbc = use_pbc
        self.regress_forces = regress_forces
        self.use_equivariant_block= use_equivariant_block # 是否使用等变网络
        self.use_CDR3_att_block = use_CDR3_att_block #是否使用CDR3与全局的等变与不变注意力网络
        self.use_invariant_block= use_invariant_block # 是否使用等变网络
        self.use_invariant_block_for_tran = use_invariant_block_for_tran
        self.use_deep_net = use_deep_net
        self.use_KL_loss = use_KL_loss # use kl loss
        self.otf_graph = otf_graph
        self.max_neighbors = max_neighbors
        self.max_radius = max_radius
        self.cutoff = max_radius
        self.max_num_elements = max_num_elements
        self.max_num_atom_names = max_num_atom_names
        self.max_num_residues = max_num_residues

        self.num_layers = num_layers
        self.sphere_channels = sphere_channels
        self.attn_hidden_channels = attn_hidden_channels
        self.num_heads = num_heads
        self.attn_alpha_channels = attn_alpha_channels
        self.attn_value_channels = attn_value_channels
        self.ffn_hidden_channels = ffn_hidden_channels
        self.norm_type = norm_type

        self.lmax_list = lmax_list
        self.mmax_list = mmax_list
        self.grid_resolution = grid_resolution

        self.num_sphere_samples = num_sphere_samples

        self.edge_channels = edge_channels
        self.use_atom_edge_embedding = use_atom_edge_embedding
        self.share_atom_edge_embedding = share_atom_edge_embedding
        if self.share_atom_edge_embedding:
            assert self.use_atom_edge_embedding
            self.block_use_atom_edge_embedding = False
        else:
            self.block_use_atom_edge_embedding = self.use_atom_edge_embedding
        self.use_m_share_rad = use_m_share_rad
        self.distance_function = distance_function
        self.num_distance_basis = num_distance_basis

        self.attn_activation = attn_activation
        self.use_s2_act_attn = use_s2_act_attn
        self.use_attn_renorm = use_attn_renorm
        self.ffn_activation = ffn_activation
        self.use_gate_act = use_gate_act
        self.use_grid_mlp = use_grid_mlp
        self.use_sep_s2_act = use_sep_s2_act

        self.alpha_drop = alpha_drop
        self.drop_path_rate = drop_path_rate
        self.proj_drop = proj_drop

        self.weight_init = weight_init
        assert self.weight_init in ['normal', 'uniform']

        self.device = 'cpu'  # torch.cuda.current_device()

        self.grad_forces = False
        self.num_resolutions = len(self.lmax_list)
        self.sphere_channels_all = self.num_resolutions * self.sphere_channels

        # Weights for message initialization
        self.sphere_embedding = nn.Embedding(self.max_num_elements, self.sphere_channels_all)
        self.element_sphere_embedding = nn.Embedding(self.max_num_elements, self.sphere_channels_all)
        self.atom_name_sphere_embedding = nn.Embedding(self.max_num_atom_names, self.sphere_channels_all)
        self.resid_sphere_embedding = nn.Embedding(self.max_num_residues, self.sphere_channels_all)

        # Initialize the function used to measure the distances between atoms
        assert self.distance_function in [
            'gaussian',
        ]
        if self.distance_function == 'gaussian':
            self.distance_expansion = GaussianSmearing(
                0.0,
                self.cutoff,
                600,
                2.0,
            )
            # self.distance_expansion = GaussianRadialBasisLayer(num_basis=self.num_distance_basis, cutoff=self.max_radius)
        else:
            raise ValueError

        # Initialize the sizes of radial functions (input channels and 2 hidden channels)
        self.edge_channels_list = [int(self.distance_expansion.num_output)] + [self.edge_channels] * 2

        # Initialize atom edge embedding
        if self.share_atom_edge_embedding and self.use_atom_edge_embedding:
            self.source_embedding = nn.Embedding(self.max_num_elements, self.edge_channels_list[-1])
            self.target_embedding = nn.Embedding(self.max_num_elements, self.edge_channels_list[-1])
            self.edge_channels_list[0] = self.edge_channels_list[0] + 2 * self.edge_channels_list[-1]
        else:
            self.source_embedding, self.target_embedding = None, None

        # Initialize the module that compute WignerD matrices and other values for spherical harmonic calculations
        self.SO3_rotation = nn.ModuleList()
        for i in range(self.num_resolutions):
            self.SO3_rotation.append(SO3_Rotation(self.lmax_list[i]))

        # Initialize conversion between degree l and order m layouts
        self.mappingReduced = CoefficientMappingModule(self.lmax_list, self.mmax_list)

        # Initialize the transformations between spherical and grid representations
        self.SO3_grid = ModuleListInfo('({}, {})'.format(max(self.lmax_list), max(self.lmax_list)))
        for l in range(max(self.lmax_list) + 1):
            SO3_m_grid = nn.ModuleList()
            for m in range(max(self.lmax_list) + 1):
                SO3_m_grid.append(
                    SO3_Grid(
                        l,
                        m,
                        resolution=self.grid_resolution,
                        normalization='component'
                    )
                )
            self.SO3_grid.append(SO3_m_grid)

        # Edge-degree embedding
        self.edge_degree_embedding = EdgeDegreeEmbedding(
            3*self.sphere_channels,
            self.lmax_list,
            self.mmax_list,
            self.SO3_rotation,
            self.mappingReduced,
            self.max_num_elements,
            self.edge_channels_list,
            self.block_use_atom_edge_embedding,
            rescale_factor=_AVG_DEGREE
        )

        # Initialize the blocks for each layer of EquiformerV2
        self.blocks = nn.ModuleList()
        for i in range(self.num_layers):
            block = TransBlockV2(
                3*self.sphere_channels,
                self.attn_hidden_channels,
                self.num_heads,
                self.attn_alpha_channels,
                self.attn_value_channels,
                self.ffn_hidden_channels,
                3*self.sphere_channels,
                self.lmax_list,
                self.mmax_list,
                self.SO3_rotation,
                self.mappingReduced,
                self.SO3_grid,
                self.max_num_elements,
                self.edge_channels_list,
                self.block_use_atom_edge_embedding,
                self.use_m_share_rad,
                self.attn_activation,
                self.use_s2_act_attn,
                self.use_attn_renorm,
                self.ffn_activation,
                self.use_gate_act,
                self.use_grid_mlp,
                self.use_sep_s2_act,
                self.norm_type,
                self.alpha_drop,
                self.drop_path_rate,
                self.proj_drop
            )
            self.blocks.append(block)

        # Output blocks for energy and forces
        self.norm = get_normalization_layer(self.norm_type, lmax=max(self.lmax_list), num_channels=3*self.sphere_channels)
        self.energy_block = FeedForwardNetwork(
            3*self.sphere_channels,
            self.ffn_hidden_channels,
            1,
            self.lmax_list,
            self.mmax_list,
            self.SO3_grid,
            self.ffn_activation,
            self.use_gate_act,
            self.use_grid_mlp,
            self.use_sep_s2_act
        )

        if self.use_invariant_block: # 该网络用于distmap 预测
            # 将上一层网络输出的节点特征，进行广播拼接之后，再输入到不变网络中，直接预测distmap。
            self.invariant_block = FeedForwardNetwork(
                3*self.sphere_channels,
                self.ffn_hidden_channels,
                8,   # can be modified, GNNRefine是采用的softmax，参考使用。非等距分布
                self.lmax_list,
                self.mmax_list,
                self.SO3_grid,
                self.ffn_activation,
                self.use_gate_act,
                self.use_grid_mlp,
                self.use_sep_s2_act
            )

            self.invariant_block_output_MLP = nn.Sequential(
                nn.Linear(8 * 2, 128),
                nn.ReLU(),
                nn.Linear(128, 38)
            )

        if self.use_invariant_block_for_tran: # 是否使用不变网络预测tran
            if self.use_deep_net:
                self.invariant_block_for_tran = FeedForwardNetwork(
                    3*self.sphere_channels,
                    self.ffn_hidden_channels,
                    3*self.sphere_channels,   # can be modified, GNNRefine是采用的softmax，参考使用。非等距分布
                    self.lmax_list,
                    self.mmax_list,
                    self.SO3_grid,
                    self.ffn_activation,
                    self.use_gate_act,
                    self.use_grid_mlp,
                    self.use_sep_s2_act
                )
                # 加深不变头
                self.invariant_block_for_tran2 = FeedForwardNetwork(
                    3*self.sphere_channels,
                    self.ffn_hidden_channels,
                    3*self.sphere_channels,   # can be modified, GNNRefine是采用的softmax，参考使用。非等距分布
                    self.lmax_list,
                    self.mmax_list,
                    self.SO3_grid,
                    self.ffn_activation,
                    self.use_gate_act,
                    self.use_grid_mlp,
                    self.use_sep_s2_act
                )

                self.invariant_block_for_tran3 = FeedForwardNetwork(
                    3*self.sphere_channels,
                    self.ffn_hidden_channels,
                    1,   # can be modified, GNNRefine是采用的softmax，参考使用。非等距分布
                    self.lmax_list,
                    self.mmax_list,
                    self.SO3_grid,
                    self.ffn_activation,
                    self.use_gate_act,
                    self.use_grid_mlp,
                    self.use_sep_s2_act
                )
            else:
                self.invariant_block_for_tran = FeedForwardNetwork(
                    3*self.sphere_channels,
                    self.ffn_hidden_channels,
                    1,   # can be modified, GNNRefine是采用的softmax，参考使用。非等距分布
                    self.lmax_list,
                    self.mmax_list,
                    self.SO3_grid,
                    self.ffn_activation,
                    self.use_gate_act,
                    self.use_grid_mlp,
                    self.use_sep_s2_act
                )

        if self.use_KL_loss:
            self.equivariant_mu = SO2EquivariantGraphAttention(
                3*self.sphere_channels,
                self.attn_hidden_channels,
                self.num_heads,
                self.attn_alpha_channels,
                self.attn_value_channels,
                3,
                self.lmax_list,
                self.mmax_list,
                self.SO3_rotation,
                self.mappingReduced,
                self.SO3_grid,
                self.max_num_elements,
                self.edge_channels_list,
                self.block_use_atom_edge_embedding,
                self.use_m_share_rad,
                self.attn_activation,
                self.use_s2_act_attn,
                self.use_attn_renorm,
                self.use_gate_act,
                self.use_sep_s2_act,
                alpha_drop=0.0
            )

            self.invariant_log_var = FeedForwardNetwork(
                3*self.sphere_channels,
                self.ffn_hidden_channels,
                3,   # can be modified, GNNRefine是采用的softmax，参考使用。非等距分布
                self.lmax_list,
                self.mmax_list,
                self.SO3_grid,
                self.ffn_activation,
                self.use_gate_act,
                self.use_grid_mlp,
                self.use_sep_s2_act
            )

            self.invariant_log_weights = FeedForwardNetwork(
                3*self.sphere_channels,
                self.ffn_hidden_channels,
                3,   # can be modified, GNNRefine是采用的softmax，参考使用。非等距分布
                self.lmax_list,
                self.mmax_list,
                self.SO3_grid,
                self.ffn_activation,
                self.use_gate_act,
                self.use_grid_mlp,
                self.use_sep_s2_act
            )
           
        if self.use_deep_net:
            self.equivariant_block = SO2EquivariantGraphAttention(
                3*self.sphere_channels,
                self.attn_hidden_channels,
                self.num_heads,
                self.attn_alpha_channels,
                self.attn_value_channels,
                3*self.sphere_channels,
                self.lmax_list,
                self.mmax_list,
                self.SO3_rotation,
                self.mappingReduced,
                self.SO3_grid,
                self.max_num_elements,
                self.edge_channels_list,
                self.block_use_atom_edge_embedding,
                self.use_m_share_rad,
                self.attn_activation,
                self.use_s2_act_attn,
                self.use_attn_renorm,
                self.use_gate_act,
                self.use_sep_s2_act,
                alpha_drop=0.0
            )
            self.equivariant_block2 = SO2EquivariantGraphAttention(
                3*self.sphere_channels,
                self.attn_hidden_channels,
                self.num_heads,
                self.attn_alpha_channels,
                self.attn_value_channels,
                3*self.sphere_channels,
                self.lmax_list,
                self.mmax_list,
                self.SO3_rotation,
                self.mappingReduced,
                self.SO3_grid,
                self.max_num_elements,
                self.edge_channels_list,
                self.block_use_atom_edge_embedding,
                self.use_m_share_rad,
                self.attn_activation,
                self.use_s2_act_attn,
                self.use_attn_renorm,
                self.use_gate_act,
                self.use_sep_s2_act,
                alpha_drop=0.0
            )
            self.equivariant_block3 = SO2EquivariantGraphAttention(
                3*self.sphere_channels,
                self.attn_hidden_channels,
                self.num_heads,
                self.attn_alpha_channels,
                self.attn_value_channels,
                1,
                self.lmax_list,
                self.mmax_list,
                self.SO3_rotation,
                self.mappingReduced,
                self.SO3_grid,
                self.max_num_elements,
                self.edge_channels_list,
                self.block_use_atom_edge_embedding,
                self.use_m_share_rad,
                self.attn_activation,
                self.use_s2_act_attn,
                self.use_attn_renorm,
                self.use_gate_act,
                self.use_sep_s2_act,
                alpha_drop=0.0
            )

        else:
            self.equivariant_block = SO2EquivariantGraphAttention(
                3*self.sphere_channels,
                self.attn_hidden_channels,
                self.num_heads,
                self.attn_alpha_channels,
                self.attn_value_channels,
                1,
                self.lmax_list,
                self.mmax_list,
                self.SO3_rotation,
                self.mappingReduced,
                self.SO3_grid,
                self.max_num_elements,
                self.edge_channels_list,
                self.block_use_atom_edge_embedding,
                self.use_m_share_rad,
                self.attn_activation,
                self.use_s2_act_attn,
                self.use_attn_renorm,
                self.use_gate_act,
                self.use_sep_s2_act,
                alpha_drop=0.0
            )

        if self.regress_forces: 
            self.force_block = SO2EquivariantGraphAttention(
                3*self.sphere_channels,
                self.attn_hidden_channels,
                self.num_heads,
                self.attn_alpha_channels,
                self.attn_value_channels,
                1,
                self.lmax_list,
                self.mmax_list,
                self.SO3_rotation,
                self.mappingReduced,
                self.SO3_grid,
                self.max_num_elements,
                self.edge_channels_list,
                self.block_use_atom_edge_embedding,
                self.use_m_share_rad,
                self.attn_activation,
                self.use_s2_act_attn,
                self.use_attn_renorm,
                self.use_gate_act,
                self.use_sep_s2_act,
                alpha_drop=0.0
            )

        self.apply(self._init_weights)
        self.apply(self._uniform_init_rad_func_linear_weights)

        if self.use_CDR3_att_block:
            # 初始化等变CDR3与全局网络
            self.EquivariantCDR3Processor = EquivariantCDR3Processor(3)
            self.InvariantCDR3Processor = InvariantCDR3Processor(1)
    
    # 应该再获得一个distmap的mask，在数据处理阶段，就根据距离cutoff，获取一部分的pair进去到模型
    def output_invariant_block(self, x): 
        # x (N,F)
        # 先输入到self.invariant_block，后面再跟MLP
        invariant_block_output = self.invariant_block(x)
        invariant_block_output_embedding = invariant_block_output.embedding.narrow(1, 0, 1)
        invariant_block_output_embedding = invariant_block_output_embedding.squeeze(1)
        N = invariant_block_output_embedding.size(0)
        atom_i = invariant_block_output_embedding.unsqueeze(1).expand(N, N, invariant_block_output_embedding.size(-1))
        atom_j = invariant_block_output_embedding.unsqueeze(0).expand(N, N, invariant_block_output_embedding.size(-1))
        pair_feat = torch.cat([atom_i, atom_j], dim=-1)
        mask = torch.triu(torch.ones(N, N), diagonal=1).bool()
        unique_pair_feat = pair_feat[mask]
        
        distmap_pred = self.invariant_block_output_MLP(unique_pair_feat)
        
        #[M, 38], M=(N-1)N/2
        return distmap_pred


    @conditional_grad(torch.enable_grad())
    def forward(self, data):
        # print(data)
        self.batch_size = len(data.n_nodes)
        self.dtype = data.pos.dtype
        self.device = data.pos.device
        # print(self.device)

        atomic_numbers = data.atomic_numbers.long()
        atom_numbers = data.atom_numbers.long()
        resid = data.resid.long()
        num_atoms = len(atomic_numbers)
        # pos = data.pos

        # to adapt with ocpmodel BaseModel, we should modify some attributes
        data.natoms = data.n_nodes

        (
            edge_index,
            edge_distance,
            edge_distance_vec,
            cell_offsets,
            _,  # cell offset distances
            neighbors,
        ) = self.generate_graph(data)

        ###############################################################
        # Initialize data structures
        ###############################################################

        # Compute 3x3 rotation matrix per edge
        edge_rot_mat = self._init_edge_rot_mat(
            data, edge_index, edge_distance_vec
        )

        # Initialize the WignerD matrices and other values for spherical harmonic calculations
        for i in range(self.num_resolutions):
            self.SO3_rotation[i].set_wigner(edge_rot_mat)

        ###############################################################
        # Initialize node embeddings
        ###############################################################

        # Init per node representations using an atomic number based embedding
        offset = 0
        x = SO3_Embedding( # there x dim is [atom, 25, 96], 25 is from (lmax_list[0] + 1) ** 2
            num_atoms,
            self.lmax_list,
            3*self.sphere_channels,
            self.device,
            self.dtype,
        )


        offset_res = 0
        offset = 0
        # Initialize the l = 0, m = 0 coefficients for each resolution
        for i in range(self.num_resolutions):
            if self.num_resolutions == 1:
                x.embedding[:, offset_res, :] = torch.cat([self.element_sphere_embedding(atomic_numbers),
                                                           self.atom_name_sphere_embedding(atom_numbers),
                                                           self.resid_sphere_embedding(resid)], dim=-1)
                # x.embedding[:, offset_res, :] = self.sphere_embedding(atomic_numbers)
            else:
                x.embedding[:, offset_res, :] = torch.cat([self.element_sphere_embedding(atomic_numbers)[:, offset: offset + self.sphere_channels],
                                                           self.atom_name_sphere_embedding(atom_numbers)[:, offset: offset + self.sphere_channels],
                                                           self.resid_sphere_embedding(resid)[:, offset: offset + self.sphere_channels]], dim=-1)
                # x.embedding[:, offset_res, :] = self.sphere_embedding(
                #     atomic_numbers
                # )[:, offset: offset + self.sphere_channels]
            offset = offset + self.sphere_channels
            offset_res = offset_res + int((self.lmax_list[i] + 1) ** 2)

        # Edge encoding (distance and atom edge)
        edge_distance = self.distance_expansion(edge_distance)
        if self.share_atom_edge_embedding and self.use_atom_edge_embedding:
            source_element = atomic_numbers[edge_index[0]]  # Source atom atomic number
            target_element = atomic_numbers[edge_index[1]]  # Target atom atomic number
            source_embedding = self.source_embedding(source_element)
            target_embedding = self.target_embedding(target_element)
            edge_distance = torch.cat((edge_distance, source_embedding, target_embedding), dim=1)

        # Edge-degree embedding
        edge_degree = self.edge_degree_embedding(
            atomic_numbers,
            edge_distance,
            edge_index)
        # print(torch.sum(x.embedding), torch.sum(edge_degree.embedding))
        # re_edge_emb = edge_degree.embedding
        x.embedding = x.embedding + edge_degree.embedding

        ###############################################################
        # Update spherical node embeddings
        ###############################################################

        for i in range(self.num_layers):
            x = self.blocks[i](
                x,  # SO3_Embedding
                atomic_numbers,
                edge_distance,
                edge_index,
                batch=data.batch  # for GraphDropPath
            )

        # Final layer norm
        x.embedding = self.norm(x.embedding)


        ###############################################################
        # score estimation
        ###############################################################
        # if the output is equivariant 
        if self.use_deep_net:
            equivariant_tran = self.equivariant_block(
                x,
                atomic_numbers,
                edge_distance,
                edge_index)
            equivariant_tran = self.equivariant_block2(
                equivariant_tran,
                atomic_numbers,
                edge_distance,
                edge_index)
            equivariant_tran = self.equivariant_block3(
                equivariant_tran,
                atomic_numbers,
                edge_distance,
                edge_index)
        else:
            equivariant_tran = self.equivariant_block(
                x,
                atomic_numbers,
                edge_distance,
                edge_index)

        equivariant_tran = equivariant_tran.embedding.narrow(1, 1, 3)
        atom_embedding = equivariant_tran.view(-1, 3)


        # return atom_embedding

        # ###############################################################
        # # Force estimation
        # ###############################################################
        if self.use_KL_loss:
            equivariant_mu_em = self.equivariant_mu(
                    x,
                    atomic_numbers,
                    edge_distance,
                    edge_index)

            equivariant_mu_em = equivariant_mu_em.embedding.narrow(1, 1, 3)
            atom_mu = equivariant_mu_em.view(-1, 3, 3)

            invariant_output_log_var = self.invariant_log_var(x)
            atom_log_var = invariant_output_log_var.embedding.narrow(1, 0, 1)

            invariant_output_log_weights = self.invariant_log_weights(x)
            atom_log_weight = invariant_output_log_weights.embedding.narrow(1, 0, 1).squeeze(1)
        else:
            atom_mu = None
            atom_log_var = None
            atom_log_weight= None

        if self.use_invariant_block_for_tran:
            if self.use_deep_net:
                pred_tran = self.invariant_block_for_tran(x)
                pred_tran = self.invariant_block_for_tran2(pred_tran)
                pred_tran = self.invariant_block_for_tran3(pred_tran)
                pred_tran = pred_tran.embedding.narrow(1, 0, 1).squeeze(1)
            else:  
                pred_tran = self.invariant_block_for_tran(x)
                pred_tran = pred_tran.embedding.narrow(1, 0, 1).squeeze(1)
            pred_tran_merget = pred_tran.clone()
        else:
            pred_tran = None
            pred_tran_merget = None



        # 创建可写副本
        atom_embedding_merged = atom_embedding.clone() 
        
        if self.use_CDR3_att_block:
            # 在等变头使用 CDR3和全局的等变注意力网络
            cdr3_mask = (data.cdr3_mask_backbone_atom == 1).bool()
            equ_CDR3_atom_embedding = self.EquivariantCDR3Processor(atom_embedding[cdr3_mask].unsqueeze(0), atom_embedding.unsqueeze(0)).squeeze(0)
            inv_CDR3_atom_embedding = self.InvariantCDR3Processor(pred_tran[cdr3_mask].unsqueeze(0),pred_tran.unsqueeze(0)).squeeze(0)

             
            # 索引赋值操作,并返回用于计算loss   
            atom_embedding_merged[cdr3_mask] = equ_CDR3_atom_embedding  
            pred_tran_merget[cdr3_mask] = inv_CDR3_atom_embedding  

            
        else:
            equ_CDR3_atom_embedding = None
            inv_CDR3_atom_embedding =None


        if self.use_invariant_block: # distmap loss
            pred_distmap = self.output_invariant_block(x)
            return pred_distmap, atom_embedding_merged, atom_mu, atom_log_var, atom_log_weight, pred_tran_merget, equ_CDR3_atom_embedding, inv_CDR3_atom_embedding
        else:
            return None, atom_embedding_merged, atom_mu, atom_log_var, atom_log_weight, pred_tran_merget, equ_CDR3_atom_embedding, inv_CDR3_atom_embedding
        # if self.regress_forces:
        #     forces = self.force_block(x,
        #                               atomic_numbers,
        #                               edge_distance,
        #                               edge_index)
        #     forces = forces.embedding.narrow(1, 1, 3)
        #     forces = forces.view(-1, 3)
        #
        # if not self.regress_forces:
        #     return energy
        # else:
        #     return energy, forces

    # Initialize the edge rotation matrics
    def _init_edge_rot_mat(self, data, edge_index, edge_distance_vec):
        return init_edge_rot_mat(edge_distance_vec)

    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    def _init_weights(self, m):
        if (isinstance(m, torch.nn.Linear)
                or isinstance(m, SO3_LinearV2)
        ):
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
            if self.weight_init == 'normal':
                std = 1 / math.sqrt(m.in_features)
                torch.nn.init.normal_(m.weight, 0, std)

        elif isinstance(m, torch.nn.LayerNorm):
            torch.nn.init.constant_(m.bias, 0)
            torch.nn.init.constant_(m.weight, 1.0)

    def _uniform_init_rad_func_linear_weights(self, m):
        if (isinstance(m, RadialFunction)):
            m.apply(self._uniform_init_linear_weights)

    def _uniform_init_linear_weights(self, m):
        if isinstance(m, torch.nn.Linear):
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
            std = 1 / math.sqrt(m.in_features)
            torch.nn.init.uniform_(m.weight, -std, std)

    @torch.jit.ignore
    def no_weight_decay(self):
        no_wd_list = []
        named_parameters_list = [name for name, _ in self.named_parameters()]
        for module_name, module in self.named_modules():
            if (isinstance(module, torch.nn.Linear)
                    or isinstance(module, SO3_LinearV2)
                    or isinstance(module, torch.nn.LayerNorm)
                    or isinstance(module, EquivariantLayerNormArray)
                    or isinstance(module, EquivariantLayerNormArraySphericalHarmonics)
                    or isinstance(module, EquivariantRMSNormArraySphericalHarmonics)
                    or isinstance(module, EquivariantRMSNormArraySphericalHarmonicsV2)
                    or isinstance(module, GaussianRadialBasisLayer)):
                for parameter_name, _ in module.named_parameters():
                    if (isinstance(module, torch.nn.Linear)
                            or isinstance(module, SO3_LinearV2)
                    ):
                        if 'weight' in parameter_name:
                            continue
                    global_parameter_name = module_name + '.' + parameter_name
                    assert global_parameter_name in named_parameters_list
                    no_wd_list.append(global_parameter_name)
        return set(no_wd_list)