import math

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F

def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


def conv_branch_init(conv, branches):
    weight = conv.weight
    n = weight.size(0)
    k1 = weight.size(1)
    k2 = weight.size(2)
    nn.init.normal_(weight, 0, math.sqrt(2. / (n * k1 * k2 * branches)))
    if conv.bias is not None:
        nn.init.constant_(conv.bias, 0)


def conv_init(conv):
    if conv.weight is not None:
        nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    if conv.bias is not None:
        nn.init.constant_(conv.bias, 0)


def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        if hasattr(m, 'weight'):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
        if hasattr(m, 'bias') and m.bias is not None and isinstance(m.bias, torch.Tensor):
            nn.init.constant_(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        if hasattr(m, 'weight') and m.weight is not None:
            m.weight.data.normal_(1.0, 0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            m.bias.data.fill_(0)
class DynamicHYPERGC(nn.Module):
    def __init__(self, in_channels, out_channels, A, num_subset=8, rel_reduction=4, topk=3):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_subset = num_subset
        self.topk = topk

        self.PA = nn.Parameter(torch.from_numpy(A.astype(np.float32)), requires_grad=False)
        self.edge_importance = nn.Parameter(torch.ones(self.PA.shape))

        mid_in_channels = in_channels // num_subset
        self.hidden_channels = mid_in_channels // rel_reduction
        self.relu = nn.ReLU()

        # feature projections (you used 2*C because you concat topology node-emb)
        self.to_V = nn.Conv1d(2 * in_channels, num_subset * self.hidden_channels, kernel_size=1, groups=num_subset)
        self.to_W = nn.Sequential(
            nn.Conv1d(2 * in_channels, num_subset * self.hidden_channels, kernel_size=1, groups=num_subset),
            nn.LeakyReLU(),
            nn.Conv1d(num_subset * self.hidden_channels, num_subset, kernel_size=1),
            nn.Tanh()
        )

        # --------- shortest-path hop distance (D) ----------
        h1 = A.sum(0)
        h1[h1 != 0] = 1

        h = [None for _ in range(A.shape[-1])]
        h[0] = np.eye(A.shape[-1])
        h[1] = h1
        self.hops = 0 * h[0]

        for i in range(2, A.shape[-1]):
            h[i] = h[i - 1] @ h1.transpose(0, 1)
            h[i][h[i] != 0] = 1

        for i in range(A.shape[-1] - 1, 0, -1):
            if np.any(h[i] - h[i - 1]):
                h[i] = h[i] - h[i - 1]
                self.hops += i * h[i]

        self.hops = torch.tensor(self.hops).long()  # (V,V)

        # --------- learned topology priors ----------
        # node topology embedding (used to enrich node features)
        max_hop = int(self.hops.max().item())
        self.rpe = nn.Parameter(torch.zeros((max_hop + 1, in_channels)))  # (D+1, C)

        # OPTION 1: hop-distance bias added to pairwise logits (does NOT forbid long-range)
        self.hop_bias = nn.Parameter(torch.zeros(num_subset, max_hop + 1))  # (S, D+1)

        # feature scale for similarity
        self.sigma_h = nn.Parameter(torch.tensor(2.0))

        # motifs weights
        self.triangle_weight = nn.Parameter(torch.tensor(0.5))
        self.beta_wedge = nn.Parameter(torch.tensor(0.5))


        # conv + residual
        self.alpha = nn.Parameter(torch.ones(1))
        self.conv_d = nn.Conv2d(in_channels, out_channels, kernel_size=1, groups=num_subset)
        self.bn = nn.BatchNorm2d(out_channels)
        self.tau = 0.5

        if in_channels != out_channels:
            self.down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.down = lambda x: x

        nn.init.kaiming_normal_(self.conv_d.weight, mode="fan_out")
        if self.conv_d.bias is not None:
            nn.init.constant_(self.conv_d.bias, 0)
        nn.init.constant_(self.bn.weight, 1e-6)
        nn.init.constant_(self.bn.bias, 0)

        # keep your project init helpers if they exist
        conv_init(self.to_W[0])
        conv_init(self.to_W[2])

        # small init so hop prior doesn't dominate early
        nn.init.normal_(self.hop_bias, mean=0.0, std=1e-3)

    def a_norm(self, A: torch.Tensor, eps: float = 1e-8):
        d_r = torch.norm(A, p=1, dim=2, keepdim=True) + eps
        return A / d_r

    def hyper_norm(self, H, W):
        # H: (N,S,V,V) and W: (N,S,V)  -> returns adjacency (N,S,V,V)
        w = torch.diag_embed(W)  # (N,S,V,V)

        norm_w = torch.norm(H, 1, dim=2, keepdim=True) + 1e-8
        w_ = w / norm_w

        H_w = H @ w
        norm_v = torch.norm(H_w, 1, dim=3, keepdim=True) + 1e-8
        h_ = H_w / norm_v

        A = h_ @ w_ @ H.transpose(3, 2)
        return A

    def forward(self, x):
        # x: (N,C,T,V)
        N, C, T, V = x.shape

        A = self.PA.to(x.device)
        A = self.edge_importance.to(x.device) * A
        A = self.a_norm(A)

        hops = self.hops.to(x.device)  # (V,V)
        eye_mask = torch.eye(V, device=x.device).bool()

        # ===== node topology embedding (enrich node features) =====
        # rpe[hops]: (V,V,C). average over neighbors -> (V,C)
        pos_emb = self.rpe[hops].mean(dim=1)                         # (V,C)
        pos_emb_node = pos_emb.T.unsqueeze(0).expand(N, -1, -1)      # (N,C,V)

        # ===== node features =====
        t_x = x.mean(2)                                              # (N,C,V)
        t_x_enriched = torch.cat([t_x, pos_emb_node], dim=1)          # (N,2C,V)

        # ===== feature-based pair logits =====
        v_x = self.to_V(t_x_enriched)                                 # (N,S*Hc,V)
        dis_v_x = v_x.view(N, self.num_subset, self.hidden_channels, V).permute(0, 1, 3, 2).contiguous()
        distance_x = torch.cdist(dis_v_x, dis_v_x)                    # (N,S,V,V)

        sigma = F.softplus(self.sigma_h) + 1e-6
        logit_feat = -(distance_x ** 2) / (2.0 * sigma ** 2)          # (N,S,V,V)

        # ===== OPTION 1: add learned hop-distance bias to logits =====
        # hop_bias[:, hops] -> (S,V,V) -> (N,S,V,V)
        logit_hop = self.hop_bias[:, hops].unsqueeze(0).expand(N, -1, -1, -1)

        logits = logit_feat + logit_hop                               # (N,S,V,V)

        # turn into soft adjacency weights (still allows long-range if logits say so)
        FCG = torch.softmax(logits, dim=-1)                            # (N,S,V,V)
        FCG = FCG.masked_fill(eye_mask, 0.0)

        # ===== optional sparsify (keeps motif ops stable) =====
        K0 = max(1, int(V * 0.1))
        vals0, _ = torch.topk(FCG, K0, dim=-1, largest=True, sorted=False)
        thr = vals0.mean(dim=-1, keepdim=True)
        FCG = torch.where(FCG < thr, torch.zeros_like(FCG), FCG)

        # ===== motifs (second/third order) =====

        W2_triangle = (FCG @ FCG) * FCG
        W2_wedge = (FCG @ FCG) * (1 - (FCG > 0).float())


        W2 = (self.triangle_weight * W2_triangle + self.beta_wedge * W2_wedge)

        # ===== incidence from motifs =====
        topk_v, topk_indices = torch.topk(W2, k=self.topk, dim=-1, largest=True)
        topk_v = torch.softmax(topk_v, dim=-1)

        H_dyn = torch.zeros_like(W2)
        H_dyn.scatter_(-1, topk_indices, topk_v)                       # in-place scatter

        # ===== hypergraph -> adjacency =====
        W = self.to_W(t_x_enriched)                                     # (N,S,V)
        G = self.hyper_norm(H_dyn, W)                                   # (N,S,V,V)

        alpha = self.relu(self.alpha)
        A_fused = A.unsqueeze(0) + alpha * G

        # ===== message passing =====
        d_x = self.conv_d(x)                                            # (N,out,T,V)
        mid_out = self.out_channels // self.num_subset
        d_x = d_x.view(N, self.num_subset, mid_out, T, V)

        y = torch.einsum("nkuv,nkctv->nkctu", A_fused, d_x).contiguous()
        y = y.view(N, self.out_channels, T, V)

        y = self.bn(y)
        y = y + self.down(x)
        y = self.relu(y)
        return y, logits
 

class TemporalConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1):
        super(TemporalConv, self).__init__()
        pad = (kernel_size + (kernel_size-1) * (dilation-1) - 1) // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, 1),
            padding=(pad, 0),
            stride=(stride, 1),
            dilation=(dilation, 1),
            padding_mode='replicate')

        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class MultiScale_TemporalConv(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 dilations=[1,2,3,4],
                 residual=True,
                 residual_kernel_size=1):

        super().__init__()
        assert out_channels % (len(dilations) + 2) == 0, '# out channels should be multiples of # branches'

        # Multiple branches of temporal convolution
        self.num_branches = len(dilations) + 2
        branch_channels = out_channels // self.num_branches
        branch_mid_channels = out_channels - branch_channels * (self.num_branches - 1)
        if type(kernel_size) == list:
            assert len(kernel_size) == len(dilations)
        else:
            kernel_size = [kernel_size]*len(dilations)
        # Temporal Convolution branches
        self.branches = nn.ModuleList()
        for ks, dilation in zip(kernel_size, dilations):
            self.branches.append(nn.Sequential(
                    nn.Conv2d(
                        in_channels,
                        branch_channels,
                        kernel_size=1,
                        padding=0),
                    nn.BatchNorm2d(branch_channels),
                    nn.ReLU(inplace=True),
                    TemporalConv(
                        branch_channels,
                        branch_channels,
                        kernel_size=ks,
                        stride=stride,
                        dilation=dilation),
                )
            )

        # Additional Max & 1x1 branch
        self.branches.append(nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3, 1), stride=(stride, 1), padding=(1, 0)),
            nn.BatchNorm2d(branch_channels), 
        ))

        self.branches.append(nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, padding=0, stride=(stride,1)),
            nn.BatchNorm2d(branch_channels)
        ))

        # Residual connection
        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = TemporalConv(in_channels, out_channels, kernel_size=residual_kernel_size, stride=stride)

        # initialize
        self.apply(weights_init)

    def forward(self, x):
        # Input dim: (N,C,T,V)
        res = self.residual(x)
        branch_outs = []
        for tempconv in self.branches:
            out = tempconv(x)
            branch_outs.append(out)
        out = torch.cat(branch_outs, dim=1)
        out += res
        return out


class unit_tcn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1):
        super(unit_tcn, self).__init__()
        pad = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), padding=(pad, 0),
                              stride=(stride, 1), padding_mode='replicate')

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        conv_init(self.conv)
        bn_init(self.bn, 1)

    def forward(self, x):
        x = self.bn(self.conv(x))
        return x


class TCN_GCN_unit(nn.Module):
    def __init__(self, in_channels, out_channels, num_point, hyper_joints, A, stride=1, residual=True, kernel_size=5, dilations=[1, 2], hyper=True):
        super(TCN_GCN_unit, self).__init__()
        self.gcn1 = DynamicHYPERGC(in_channels, out_channels,  A)
        self.tcn1 = MultiScale_TemporalConv(out_channels, out_channels, kernel_size=kernel_size, stride=stride, dilations=dilations,
                                        residual=False)
        self.relu = nn.ReLU(inplace=True)
        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = unit_tcn(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x):
        y , graph = self.gcn1(x)
        #print(h_x.shape)
        y = self.relu(self.tcn1(y) + self.residual(x))
        return y , graph

class Refine_unit(nn.Module):
    def __init__(self, out_channels,n_prototype  , dropout=0.1): 
        super(Refine_unit, self).__init__()
        self.num_subset = out_channels
        self.n_prototype = n_prototype
        

        self.spatial_prototype = nn.Embedding(self.n_prototype, self.num_subset)
        self.fc_spatial = nn.Sequential(
            nn.Linear(self.n_prototype, self.n_prototype),
            nn.ReLU(),
            nn.Linear(self.n_prototype, self.num_subset),
            nn.ReLU()
        )
        self.softmax = torch.softmax
        self.dropout = nn.Dropout(dropout)
        



    def forward(self, x):
        
        N= x.size(0)
        sp = self.spatial_prototype.weight.unsqueeze(0).expand(N, -1, -1).permute(0,2,1)
        sp_select = torch.einsum('ncj,ncv->njv', x, sp)  # n, V*E, 100
        
        corr = self.softmax(sp_select, dim=-1)
        

        sp = self.fc_spatial(corr).permute(0,2,1)
        #print(sp.shape)


        return self.dropout(sp)
class Model(nn.Module):
    def __init__(self, num_class=60, num_point=25, num_person=2, graph=None, graph_args=dict(), in_channels=3, hyper_joints=0,
                 drop_out=0):
        super(Model, self).__init__()

        if graph is None:
            raise ValueError()
        else:
            Graph = import_class(graph)
            self.graph = Graph(hyper_joints, **graph_args)

        A = self.graph.A
        self.num_class = num_class
        self.num_point = num_point
        self.embedding_channels = 128

        self.data_bn = nn.BatchNorm1d(num_person * self.embedding_channels * num_point)
        self.to_joint_embedding = nn.Linear(in_channels, self.embedding_channels)
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_point, self.embedding_channels))
        self.tanh = nn.Tanh()
        self.num_subset = 8

        self.l1 = TCN_GCN_unit(self.embedding_channels, self.embedding_channels, num_point, hyper_joints, A)
        self.l2 = TCN_GCN_unit(self.embedding_channels, self.embedding_channels, num_point, hyper_joints, A)
        self.l3 = TCN_GCN_unit(self.embedding_channels, self.embedding_channels, num_point, hyper_joints, A)
        self.l4 = TCN_GCN_unit(self.embedding_channels, self.embedding_channels * 2, num_point, hyper_joints, A, stride=2)
        self.l5 = TCN_GCN_unit(self.embedding_channels * 2, self.embedding_channels * 2, num_point, hyper_joints, A)
        self.l6 = TCN_GCN_unit(self.embedding_channels * 2, self.embedding_channels * 2, num_point, hyper_joints, A)
        self.l7 = TCN_GCN_unit(self.embedding_channels * 2, self.embedding_channels * 4, num_point, hyper_joints, A, stride=2)
        self.l8 = TCN_GCN_unit(self.embedding_channels * 4, self.embedding_channels * 4, num_point, hyper_joints, A)
        self.l9 = TCN_GCN_unit(self.embedding_channels * 4, self.embedding_channels * 4, num_point, hyper_joints, A)
        self.fc = nn.Linear(self.embedding_channels * 4, self.num_class)
        self.prn = Refine_unit(self.num_subset, n_prototype = 100)

        nn.init.normal_(self.fc.weight, 0, math.sqrt(2. / num_class))
        bn_init(self.data_bn, 1)
        if drop_out:
            self.drop_out = nn.Dropout(drop_out)
        else:
            self.drop_out = lambda x: x

    def forward(self, x):
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 2, 3, 1).contiguous()
        x = self.to_joint_embedding(x)
        x += self.pos_embedding[:, :self.num_point]
        x = self.tanh(x)
        x = x.permute(0, 1, 3, 4, 2).contiguous()

        x = x.view(N, M * V * self.embedding_channels, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, self.embedding_channels, T).permute(0, 1, 3, 4, 2).contiguous()

        x = x.view(N * M, self.embedding_channels, T, V)

        x,_= self.l1(x)
        feat_low  = x.clone()
        x1 = x
        x, _= self.l2(x)
        x, _= self.l3(x + x1)

        x, _ = self.l4(x)
        feat_mid = x.clone()

        x4 = x
        x, _ = self.l5(x)
        x, _= self.l6(x + x4)

        x, _= self.l7(x)
        feat_high = x.clone()
        x7 = x
        x, _= self.l8(x)
        x, graph = self.l9(x + x7)
        feat_fin = x.clone()
        
        # reconstruction loss

        # N*M,C,T,V
        c_new = x.size(1)
        
        #feat_fin = feat_fin.view(N, M, c_new,T_new,V).mean(1)
        #print(feat_fin.size)
        graph = graph.view(N, M, self.num_subset,-1).mean(1)
        reconstructed_graph = self.prn(graph)
        reconstructed_graph = reconstructed_graph.mean(1).view(N,-1)
        x = x.view(N, M, c_new, -1)
        x = x.mean(3).mean(1)
        x = self.drop_out(x)
        #project graph
        

        return self.fc(x), reconstructed_graph


if __name__ == '__main__':
    import os
    import sys
    import torch
    from thop import profile

    # Make imports robust: add project root (parent of this file's directory)
    THIS_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)

    device = 'cuda:0'

    model = Model(
        graph='graph.ntu_rgb_d.Graph',
        hyper_joints=3,
        graph_args={'labeling_mode': 'spatial_ensemble'}
    ).to(device)

    # PyTorch param sanity check
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Torch] Total params: {total:,} | Trainable: {trainable:,}")

    # Standard FLOPs reporting uses batch size 1
    x = torch.randn(1, 3, 64, 25, 2, device=device)

    model.eval()
    with torch.no_grad():
        try:
            macs, params = profile(model, inputs=(x,), verbose=False)
            gflops = 2.0 * macs / 1e9  # MACs -> FLOPs -> GFLOPs

            print(f"[thop] Params: {params:,}")
            print(f"[thop] MACs:   {macs/1e9:.3f} G")
            print(f"[thop] GFLOPs: {gflops:.3f} G")
        except Exception as e:
            print("thop profiling failed (likely due to unsupported ops such as einsum).")
            print("Error:", repr(e))
