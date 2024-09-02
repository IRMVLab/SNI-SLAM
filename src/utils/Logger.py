# This file is a part of SNI-SLAM.
#
# This file is a modified version of https://github.com/idiap/ESLAM/blob/main/src/utils/Logger.py
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
import os
import torch

class Logger(object):
    """
    Save checkpoints to file.

    """

    def __init__(self, sni):
        self.verbose = sni.verbose
        self.ckptsdir = sni.ckptsdir
        self.gt_c2w_list = sni.gt_c2w_list
        self.shared_decoders = sni.shared_decoders
        self.estimate_c2w_list = sni.estimate_c2w_list

        self.s_planes_xy = sni.shared_s_planes_xy
        self.s_planes_xz = sni.shared_s_planes_xz
        self.s_planes_yz = sni.shared_s_planes_yz

        self.planes_xy = sni.shared_planes_xy
        self.planes_xz = sni.shared_planes_xz
        self.planes_yz = sni.shared_planes_yz

        self.c_planes_xy = sni.shared_c_planes_xy
        self.c_planes_xz = sni.shared_c_planes_xz
        self.c_planes_yz = sni.shared_c_planes_yz

    def log(self, idx, keyframe_list):
        path = os.path.join(self.ckptsdir, '{:05d}.tar'.format(idx))
        torch.save({
            'decoder_state_dict': self.shared_decoders.state_dict(),

            's_planes_xy': self.s_planes_xy,
            's_planes_xz': self.s_planes_xz,
            's_planes_yz': self.s_planes_yz,
            'planes_xy': self.planes_xy,
            'planes_xz': self.planes_xz,
            'planes_yz': self.planes_yz,
            'c_planes_xy': self.c_planes_xy,
            'c_planes_xz': self.c_planes_xz,
            'c_planes_yz': self.c_planes_yz,

            'gt_c2w_list': self.gt_c2w_list,
            'estimate_c2w_list': self.estimate_c2w_list,
            'keyframe_list': keyframe_list,
            'idx': idx,
        }, path, _use_new_zipfile_serialization=False)

        if self.verbose:
            print('Saved checkpoints at', path)
