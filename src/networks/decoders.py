# This file is a part of SNI-SLAM.
#
# This file is a modified version of https://github.com/idiap/ESLAM/blob/main/src/networks/decoders.py
# which is covered by the following copyright and permission notice:
    #
    # Copyright 2023 ams-OSRAM AG
    #
    # Author: Mohammad Mahdi Johari <mohammad.johari@idiap.ch>
    #
    # Licensed under the Apache License, Version 2.0 (the "License");
    # you may not use this file except in compliance with the License.
    # You may obtain a copy of the License at
    #
    #     http://www.apache.org/licenses/LICENSE-2.0
    #
    # Unless required by applicable law or agreed to in writing, software
    # distributed under the License is distributed on an "AS IS" BASIS,
    # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    # See the License for the specific language governing permissions and
    # limitations under the License.

import torch
import torch.nn as nn
import torch.nn.functional as F
from src.common import normalize_3d_coordinate

class Decoders(nn.Module):
    def __init__(self, c_dim=16, hidden_dim=32, truncation=0.08, n_blocks=2, learnable_beta=True,
                 num_classes=25, sem_hidden_dim=256, fused_hidden_dim=16):
        super().__init__()

        self.c_dim = c_dim
        self.truncation = truncation
        self.n_blocks = n_blocks

        self.linears = nn.ModuleList(
            [nn.Linear(2 * c_dim, hidden_dim)] +
            [nn.Linear(hidden_dim, hidden_dim) for i in range(n_blocks - 1)])

        self.output_linear = nn.Linear(hidden_dim, 1)

        self.s_linears = nn.ModuleList([nn.Linear(2 * c_dim, sem_hidden_dim)] +
                                   [nn.Linear(sem_hidden_dim, sem_hidden_dim) for i in range(3 - 1)])
        self.s_output_linear = nn.Linear(sem_hidden_dim, num_classes)

        self.fused_linear = nn.ModuleList(
            [nn.Linear(2 * c_dim, fused_hidden_dim)] +
            [nn.Linear(fused_hidden_dim, fused_hidden_dim) for i in range(4 - 1)])

        self.out_sdf = nn.Linear(fused_hidden_dim, fused_hidden_dim+1)

        self.out_rgb = nn.Sequential(
            nn.Linear(fused_hidden_dim + 4 * c_dim, fused_hidden_dim),
            nn.ReLU(),
            nn.Linear(fused_hidden_dim, 3))

        if learnable_beta:
            self.beta = nn.Parameter(10 * torch.ones(1))
            self.semantic_beta = nn.Parameter(10 * torch.ones(1))
        else:
            self.beta = 10


    def sample_plane_feature(self, p_nor, planes_xy, planes_xz, planes_yz):
        vgrid = p_nor[None, :, None]
        feat = []
        for i in range(len(planes_xy)):
            xy = F.grid_sample(planes_xy[i], vgrid[..., [0, 1]], padding_mode='border', align_corners=True, mode='bilinear').squeeze().transpose(0, 1)
            xz = F.grid_sample(planes_xz[i], vgrid[..., [0, 2]], padding_mode='border', align_corners=True, mode='bilinear').squeeze().transpose(0, 1)
            yz = F.grid_sample(planes_yz[i], vgrid[..., [1, 2]], padding_mode='border', align_corners=True, mode='bilinear').squeeze().transpose(0, 1)
            feat.append(xy + xz + yz)

        feat = torch.cat(feat, dim=-1)

        return feat

    def get_raw_sdf(self, p_nor, all_planes):
        planes_xy, planes_xz, planes_yz, c_planes_xy, c_planes_xz, c_planes_yz = all_planes
        feat = self.sample_plane_feature(p_nor, planes_xy, planes_xz, planes_yz)

        h = feat
        for i, l in enumerate(self.linears):
            h = self.linears[i](h)
            h = F.relu(h, inplace=True)
        sdf = torch.tanh(self.output_linear(h)).squeeze()

        return sdf, feat

    def get_raw_semantic(self, p_nor, all_planes):
        s_planes_xy, s_planes_xz, s_planes_yz = all_planes
        s_feat = self.sample_plane_feature(p_nor, s_planes_xy, s_planes_xz, s_planes_yz)

        h = s_feat
        for i, l in enumerate(self.s_linears):
            h = self.s_linears[i](h)
            h = F.relu(h, inplace=True)
        semantic = self.s_output_linear(h)

        return semantic, s_feat

    def get_raw_sdf_rgb(self, p_nor, all_planes):
        planes_xy, planes_xz, planes_yz, c_planes_xy, c_planes_xz, c_planes_yz, s_planes_xy, s_planes_xz, s_planes_yz = all_planes

        feat = self.sample_plane_feature(p_nor, planes_xy, planes_xz, planes_yz)
        c_feat = self.sample_plane_feature(p_nor, c_planes_xy, c_planes_xz, c_planes_yz)

        s_feat = self.sample_plane_feature(p_nor, s_planes_xy, s_planes_xz, s_planes_yz)

        h = feat
        for i, l in enumerate(self.fused_linear):
            h = self.fused_linear[i](h)
            h = F.relu(h, inplace=True)
        sdf_out = self.out_sdf(h)
        sdf = torch.tanh(sdf_out[:, :1]).squeeze()
        sdf_feat = sdf_out[:, 1:]

        h = torch.cat([sdf_feat, c_feat, s_feat], dim=-1)

        rgb = torch.sigmoid(self.out_rgb(h))

        return sdf, rgb, feat, c_feat


    def forward(self, p, all_planes):
        p_shape = p.shape
        p_nor = normalize_3d_coordinate(p.clone(), self.bound)

        sdf, rgb, sdf_feat, rgb_feat = self.get_raw_sdf_rgb(p_nor, all_planes)

        semantic, semantic_feat = self.get_raw_semantic(p_nor, all_planes[6:])
        raw = torch.cat([rgb, sdf.unsqueeze(-1), semantic], dim=-1)
        plane_feat = torch.cat([rgb_feat[:, self.c_dim:], sdf_feat[:, self.c_dim:], semantic_feat[:, self.c_dim:]], dim=-1)

        raw = raw.reshape(*p_shape[:-1], -1)

        return raw, plane_feat


