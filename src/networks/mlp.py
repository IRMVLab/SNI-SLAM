# This file is a part of SNI-SLAM.

import torch
import torch.nn as nn


# Nerf positional embedding
class nerf_pos_embedding(nn.Module):
    def __init__(self, in_dim, multires, log_sampling=True):
        super().__init__()
        self.log_sampling = log_sampling
        self.include_input = True
        self.periodic_fns = [torch.sin, torch.cos]
        self.max_freq = multires-1
        self.N_freqs = multires
        self.embedding_size = multires*in_dim*2 + in_dim

    def forward(self, x):
        ray, points, _ = x.shape
        x = x.view(-1, 3)
        if self.log_sampling:
            freq_bands = 2.**torch.linspace(0., self.max_freq, steps=self.N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**self.max_freq, steps=self.N_freqs)
        output = []
        if self.include_input:
            output.append(x)
        for freq in freq_bands:
            for p_fn in self.periodic_fns:
                output.append(p_fn(x * freq))
        ret = torch.cat(output, dim=1)
        ret = ret.view(ray, points, -1)
        return ret


class encoder(nn.Module):
    def __init__(self, depth_in=3, hidden_dim=256, out_dim=128, multires=6, **kwargs):
        super().__init__()
        self.pe = nerf_pos_embedding(depth_in, multires)
        self.pts_linears = nn.ModuleList(
            [nn.Linear(self.pe.embedding_size, hidden_dim)] +
            [nn.Linear(hidden_dim, hidden_dim)] +
            [nn.Linear(hidden_dim, out_dim)] )

    def forward(self, x):
        x = self.pe(x)
        ray, points, _ = x.shape
        x = x.view(-1, self.pe.embedding_size)
        for i, l in enumerate(self.pts_linears):
            x = self.pts_linears[i](x)
        x = x.view(ray, points, -1)
        return x


class head(nn.Module):
    def __init__(self, in_dim=128, hidden_dim=256):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, in_dim)

    def forward(self, x):
        batch_size, self.in_dim, H, W = x.size()
        x = x.view(-1, self.in_dim)
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        x = x.view(batch_size, self.in_dim, H, W)
        return x

class feature_fusion(nn.Module):
    def __init__(self, dim=128, hidden_dim=256):
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.fc1 = nn.Linear(2*dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, dim).cuda()

    def self_attention(self, a_feat, b_feat, fused_feat):
        norm = torch.norm(a_feat, dim=2, keepdim=True)
        a_b = torch.matmul(b_feat, a_feat.transpose(1, 2)) / norm
        calc_softmax = torch.softmax(a_b, dim=1)
        multi_feature = torch.matmul(calc_softmax, fused_feat)
        return multi_feature

    def forward(self, semantic, depth, rgb):
        semantic_fused = self.self_attention(rgb, depth, semantic)
        x = self.self_attention(semantic_fused, depth, rgb)

        x = torch.cat((x, rgb), dim=2)
        ray, points, dim = x.size()
        x = x.view(-1, dim)

        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)

        x = x.view(ray, points, self.dim)
        return semantic_fused, x


