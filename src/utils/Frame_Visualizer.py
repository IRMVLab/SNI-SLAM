# This file is a part of SNI-SLAM.
#
# This file is a modified version of https://github.com/cvg/nice-slam/blob/master/src/utils/Visualizer.py
# which is covered by the following copyright and permission notice:
    #
    # Copyright 2022 Zihan Zhu, Songyou Peng, Viktor Larsson, Weiwei Xu, Hujun Bao, Zhaopeng Cui, Martin R. Oswald, Marc Pollefeys
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
import numpy as np
import matplotlib.pyplot as plt
from src.common import cam_pose_to_matrix
from src.tools.segmentationMetric import SegmentationMetric

class Frame_Visualizer(object):
    """
    Visualizes itermediate results, render out depth and color images.
    It can be called per iteration, which is good for debuging (to see how each tracking/mapping iteration performs).
    Args:
        freq (int): frequency of visualization.
        inside_freq (int): frequency of visualization inside each iteration.
        vis_dir (str): directory to save the visualization results.
        renderer (Renderer): renderer.
        truncation (float): truncation distance.
        verbose (bool): whether to print out the visualization results.
        device (str): device.
    """

    def __init__(self, freq, inside_freq, vis_dir, renderer, truncation, verbose, device='cuda:0', n_classes=25):
        self.freq = freq
        self.device = device
        self.vis_dir = vis_dir
        self.verbose = verbose
        self.renderer = renderer
        self.inside_freq = inside_freq
        self.truncation = truncation
        os.makedirs(f'{vis_dir}', exist_ok=True)

        self.n_classes = n_classes

        self.metric = SegmentationMetric(self.n_classes)

    def save_imgs(self, idx, iter, gt_depth, gt_color, c2w_or_camera_tensor, all_planes, decoders, gt_sem=None, est_sem=None):
        """
        Visualization of depth and color images and save to file.
        Args:
            idx (int): current frame index.
            iter (int): the iteration number.
            gt_depth (tensor): ground truth depth image of the current frame.
            gt_color (tensor): ground truth color image of the current frame.
            c2w_or_camera_tensor (tensor): camera pose, represented in 
                camera to world matrix or quaternion and translation tensor.
            all_planes (Tuple): feature planes.
            decoders (torch.nn.Module): decoders for TSDF and color.
        """
        with torch.no_grad():
            if (idx % self.freq == 0) and (iter % self.inside_freq == 0):
                gt_depth_np = gt_depth.squeeze(0).cpu().numpy()
                gt_color_np = gt_color.squeeze(0).cpu().numpy()

                if c2w_or_camera_tensor.shape[-1] > 4: ## 6od
                    c2w = cam_pose_to_matrix(c2w_or_camera_tensor.clone().detach()).squeeze()
                else:
                    c2w = c2w_or_camera_tensor.squeeze().detach()

                depth, color, semantic = self.renderer.render_img(all_planes, decoders, c2w, self.truncation,
                                                        self.device, gt_depth=gt_depth)

                depth_np = depth.detach().cpu().numpy()
                color_np = color.detach().cpu().numpy()
                depth_residual = np.abs(gt_depth_np - depth_np)
                depth_residual[gt_depth_np == 0.0] = 0.0
                color_residual = np.abs(gt_color_np - color_np)
                color_residual[gt_depth_np == 0.0] = 0.0

                if gt_sem is not None:
                    pred = semantic.detach().permute(2, 0, 1).unsqueeze(0)
                    gt = gt_sem.squeeze().unsqueeze(0).unsqueeze(0)

                    semantic_gt = torch.zeros_like(gt, dtype=torch.float32)
                    semantic_gt = torch.tile(semantic_gt, (1, self.n_classes, 1, 1))
                    for channel in range(self.n_classes):
                        channel1 = channel * 1.0
                        semantic_gt[0, channel, :, :] = gt * (gt[0, 0, :, :] == channel1)

                    self.metric.update(pred, semantic_gt)
                    pixAcc, mIoU = self.metric.get()
                    print(f"rendering: pixel acc: {pixAcc}\nmIoU: {mIoU}")
                    semantic = torch.max(semantic, 2).indices.squeeze()
                    pred_label_image = self.decode_segmap(image=semantic.detach().cpu(), nc=self.n_classes)
                    # gt_label_image = self.decode_segmap(image=gt_sem.cpu(), nc=self.n_classes)
                    gt_label_image = self.decode_segmap(image=gt_sem.squeeze().detach().cpu(), nc=self.n_classes)

                    semantic_residual = np.abs(gt_label_image - pred_label_image)
                    semantic_residual[gt_depth_np == 0.0] = 0.0

                    fig, axs = plt.subplots(3, 3)
                else:
                    fig, axs = plt.subplots(2, 3)

                fig.tight_layout()
                max_depth = np.max(gt_depth_np)

                axs[0, 0].set_title('GT')
                axs[0, 1].set_title('Generated')
                axs[0, 2].set_title('Residual')

                axs[2, 0].imshow(gt_depth_np, cmap="plasma", vmin=0, vmax=max_depth)
                axs[2, 0].set_xticks([])
                axs[2, 0].set_yticks([])
                axs[2, 1].imshow(depth_np, cmap="plasma", vmin=0, vmax=max_depth)
                axs[2, 1].set_xticks([])
                axs[2, 1].set_yticks([])
                axs[2, 2].imshow(depth_residual, cmap="plasma", vmin=0, vmax=max_depth)
                axs[2, 2].set_xticks([])
                axs[2, 2].set_yticks([])
                gt_color_np = np.clip(gt_color_np, 0, 1)
                color_np = np.clip(color_np, 0, 1)
                color_residual = np.clip(color_residual, 0, 1)
                axs[1, 0].imshow(gt_color_np, cmap="plasma")
                axs[1, 0].set_xticks([])
                axs[1, 0].set_yticks([])
                axs[1, 1].imshow(color_np, cmap="plasma")
                axs[1, 1].set_xticks([])
                axs[1, 1].set_yticks([])
                axs[1, 2].imshow(color_residual, cmap="plasma")
                axs[1, 2].set_xticks([])
                axs[1, 2].set_yticks([])

                if gt_sem is not None:
                    axs[0, 0].imshow(gt_label_image, cmap="plasma")
                    axs[0, 0].set_xticks([])
                    axs[0, 0].set_yticks([])
                    axs[0, 1].imshow(pred_label_image, cmap="plasma")
                    axs[0, 1].set_xticks([])
                    axs[0, 1].set_yticks([])
                    axs[0, 2].imshow(semantic_residual, cmap="plasma")
                    axs[0, 2].set_xticks([])
                    axs[0, 2].set_yticks([])
                    axs[0, 2].axis('off')

                plt.subplots_adjust(wspace=0, hspace=0)
                plt.savefig(f'{self.vis_dir}/{idx:05d}_{iter:04d}.jpg', bbox_inches='tight', pad_inches=0.2, dpi=300)
                plt.cla()
                plt.clf()

                if self.verbose:
                    print(f'Saved rendering visualization of color/depth image at {self.vis_dir}/{idx:05d}_{iter:04d}.jpg')

    def decode_segmap(self, image, nc=25):
        label_colors = np.array([(0, 0, 0),  # 0=background
                                 (174, 199, 232), (152, 223, 138), (31, 119, 180), (255, 187, 120), (188, 189, 34),
                                 (140, 86, 75), (255, 152, 150), (214, 39, 40), (197, 176, 213), (148, 103, 189),
                                 (196, 156, 148), (23, 190, 207), (178, 76, 76), (247, 182, 210), (66, 188, 102),
                                 (219, 219, 141), (140, 57, 197), (202, 185, 52), (51, 176, 203), (200, 54, 131),
                                 (92, 193, 61), (78, 71, 183), (172, 114, 82), (255, 127, 14), (91, 163, 138),
                                 (153, 98, 156), (140, 153, 101), (158, 218, 229), (100, 125, 154), (178, 127, 135),
                                 (120, 185, 128), (146, 111, 194), (44, 160, 44), (112, 128, 144), (96, 207, 209),
                                 (227, 119, 194), (213, 92, 176), (94, 106, 211), (82, 84, 163), (100, 85, 144),
                                 (100, 218, 200), (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128),
                                 (128, 0, 128), (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0),
                                 (192, 128, 0), (64, 0, 128)]
                                )
        r = np.zeros_like(image).astype(np.uint8)
        g = np.zeros_like(image).astype(np.uint8)
        b = np.zeros_like(image).astype(np.uint8)

        for l in range(0, nc):
            idx = image == l
            r[idx] = label_colors[l, 0]
            g[idx] = label_colors[l, 1]
            b[idx] = label_colors[l, 2]
        rgb = np.stack([r, g, b], axis=2)
        return rgb.astype(np.float32) / 255.0