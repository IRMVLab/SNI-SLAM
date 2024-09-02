# This file is a part of SNI-SLAM.


import torch
import copy
import os
import time

from torch.utils.data import DataLoader
from tqdm import tqdm

from src.common import (matrix_to_cam_pose, cam_pose_to_matrix, get_samples)
from src.utils.datasets import get_dataset
from src.utils.Frame_Visualizer import Frame_Visualizer

class Tracker(object):
    def __init__(self, cfg, args, sni):
        self.cfg = cfg
        self.args = args

        self.scale = cfg['scale']

        self.idx = sni.idx
        self.bound = sni.bound
        self.mesher = sni.mesher
        self.output = sni.output
        self.verbose = sni.verbose
        self.renderer = sni.renderer
        self.gt_c2w_list = sni.gt_c2w_list
        self.mapping_idx = sni.mapping_idx
        self.mapping_cnt = sni.mapping_cnt
        self.shared_decoders = sni.shared_decoders
        self.estimate_c2w_list = sni.estimate_c2w_list
        self.truncation = sni.truncation

        self.shared_planes_xy = sni.shared_planes_xy
        self.shared_planes_xz = sni.shared_planes_xz
        self.shared_planes_yz = sni.shared_planes_yz

        self.shared_c_planes_xy = sni.shared_c_planes_xy
        self.shared_c_planes_xz = sni.shared_c_planes_xz
        self.shared_c_planes_yz = sni.shared_c_planes_yz

        self.enable_wandb = cfg['func']['enable_wandb']
        if self.enable_wandb:
            self.wandb_run = sni.wandb_run

        self.shared_s_planes_xy = sni.shared_s_planes_xy
        self.shared_s_planes_xz = sni.shared_s_planes_xz
        self.shared_s_planes_yz = sni.shared_s_planes_yz

        self.w_semantic = cfg['tracking']['w_semantic']

        self.model_manager = sni.model_manager

        self.cam_lr_T = cfg['tracking']['lr_T']
        self.cam_lr_R = cfg['tracking']['lr_R']
        self.device = cfg['device']
        self.num_cam_iters = cfg['tracking']['iters']
        self.use_gt_pose = cfg['func']['use_gt_pose']
        self.tracking_pixels = cfg['tracking']['pixels']
        self.w_sdf_fs = cfg['tracking']['w_sdf_fs']
        self.w_sdf_center = cfg['tracking']['w_sdf_center']
        self.w_sdf_tail = cfg['tracking']['w_sdf_tail']
        self.w_depth = cfg['tracking']['w_depth']
        self.w_color = cfg['tracking']['w_color']
        self.ignore_edge_W = cfg['tracking']['ignore_edge_W']
        self.ignore_edge_H = cfg['tracking']['ignore_edge_H']
        self.const_speed_assumption = cfg['tracking']['const_speed_assumption']

        self.every_frame = cfg['mapping']['every_frame']
        self.no_vis_on_first_frame = cfg['tracking']['no_vis_on_first_frame']

        self.c_dim = cfg['model']['c_dim']

        self.prev_mapping_idx = -1
        self.frame_reader = get_dataset(cfg, args, self.scale, device=self.device)
        self.n_img = len(self.frame_reader)
        self.frame_loader = DataLoader(self.frame_reader, batch_size=1, shuffle=False,
                                       num_workers=1, pin_memory=True, prefetch_factor=2)

        self.visualizer = Frame_Visualizer(freq=cfg['tracking']['vis_freq'], inside_freq=cfg['tracking']['vis_inside_freq'],
                                           vis_dir=os.path.join(self.output, 'tracking_vis'), renderer=self.renderer,
                                           truncation=self.truncation, verbose=self.verbose, device=self.device,
                                           n_classes=cfg['model']['cnn']['n_classes'])

        self.H, self.W, self.fx, self.fy, self.cx, self.cy = sni.H, sni.W, sni.fx, sni.fy, sni.cx, sni.cy

        self.decoders = copy.deepcopy(self.shared_decoders)

        self.planes_xy = copy.deepcopy(self.shared_planes_xy)
        self.planes_xz = copy.deepcopy(self.shared_planes_xz)
        self.planes_yz = copy.deepcopy(self.shared_planes_yz)

        self.c_planes_xy = copy.deepcopy(self.shared_c_planes_xy)
        self.c_planes_xz = copy.deepcopy(self.shared_c_planes_xz)
        self.c_planes_yz = copy.deepcopy(self.shared_c_planes_yz)

        self.s_planes_xy = copy.deepcopy(self.shared_s_planes_xy)
        self.s_planes_xz = copy.deepcopy(self.shared_s_planes_xz)
        self.s_planes_yz = copy.deepcopy(self.shared_s_planes_yz)

        for p in self.decoders.parameters():
            p.requires_grad_(False)


    def sdf_losses(self, sdf, z_vals, gt_depth):
        """
        Computes the losses for a signed distance function (SDF) given its values, depth values and ground truth depth.

        Args:
        - self: instance of the class containing this method
        - sdf: a tensor of shape (R, N) representing the SDF values
        - z_vals: a tensor of shape (R, N) representing the depth values
        - gt_depth: a tensor of shape (R,) containing the ground truth depth values

        Returns:
        - sdf_losses: a scalar tensor representing the weighted sum of the free space, center, and tail losses of SDF
        """

        front_mask = torch.where(z_vals < (gt_depth[:, None] - self.truncation),
                                 torch.ones_like(z_vals), torch.zeros_like(z_vals)).bool()

        back_mask = torch.where(z_vals > (gt_depth[:, None] + self.truncation),
                                torch.ones_like(z_vals), torch.zeros_like(z_vals)).bool()

        center_mask = torch.where((z_vals > (gt_depth[:, None] - 0.4 * self.truncation)) *
                                  (z_vals < (gt_depth[:, None] + 0.4 * self.truncation)),
                                  torch.ones_like(z_vals), torch.zeros_like(z_vals)).bool()

        tail_mask = (~front_mask) * (~back_mask) * (~center_mask)

        fs_loss = torch.mean(torch.square(sdf[front_mask] - torch.ones_like(sdf[front_mask])))
        center_loss = torch.mean(torch.square(
            (z_vals + sdf * self.truncation)[center_mask] - gt_depth[:, None].expand(z_vals.shape)[center_mask]))
        tail_loss = torch.mean(torch.square(
            (z_vals + sdf * self.truncation)[tail_mask] - gt_depth[:, None].expand(z_vals.shape)[tail_mask]))

        sdf_losses = self.w_sdf_fs * fs_loss + self.w_sdf_center * center_loss + self.w_sdf_tail * tail_loss

        return sdf_losses

    def optimize_tracking(self, cam_pose, gt_color, gt_depth, batch_size, optimizer, gt_semantic=None):
        """
        Do one iteration of camera tracking. Sample pixels, render depth/color, calculate loss and backpropagation.

        Args:
            cam_pose (tensor): camera pose.
            gt_color (tensor): ground truth color image of the current frame.
            gt_depth (tensor): ground truth depth image of the current frame.
            batch_size (int): batch size, number of sampling rays.
            optimizer (torch.optim): camera optimizer.

        Returns:
            loss (float): The value of loss.
        """

        all_planes = (self.planes_xy, self.planes_xz, self.planes_yz,
                      self.c_planes_xy, self.c_planes_xz, self.c_planes_yz,
                      self.s_planes_xy, self.s_planes_xz, self.s_planes_yz)

        device = self.device
        H, W, fx, fy, cx, cy = self.H, self.W, self.fx, self.fy, self.cx, self.cy

        c2w = cam_pose_to_matrix(cam_pose)

        batch_rays_o, batch_rays_d, batch_gt_depth, batch_gt_color, _, _, batch_gt_label = get_samples(self.ignore_edge_H, H-self.ignore_edge_H,
                                                                                 self.ignore_edge_W, W-self.ignore_edge_W,
                                                                                 batch_size, H, W, fx, fy, cx, cy, c2w,
                                                                                 gt_depth, gt_color, device=device,
                                                                                 gt_label=gt_semantic, dim=self.c_dim)

        depth, color, sdf, z_vals, _, _, render_semantic = self.renderer.render_batch_ray(all_planes, self.decoders, batch_rays_d, batch_rays_o,
                                                                   self.device, self.truncation, gt_depth=batch_gt_depth)

        ## Filtering the rays for which the rendered depth error is greater than 10 times of the median depth error
        depth_error = (batch_gt_depth - depth.detach()).abs()
        error_median = depth_error.median()
        depth_mask = (depth_error < 10 * error_median)

        for param_group in optimizer.param_groups:
            for param in param_group['params']:
                param.requires_grad = True

        ## SDF losses
        sdf_loss = self.sdf_losses(sdf[depth_mask], z_vals[depth_mask], batch_gt_depth[depth_mask])
        loss = sdf_loss

        ## Color Loss
        color_loss = self.w_color * torch.square(batch_gt_color - color)[depth_mask].mean()
        loss = loss + color_loss

        ### Depth loss
        depth_loss = self.w_depth * torch.square(batch_gt_depth[depth_mask] - depth[depth_mask]).mean()
        loss = loss + depth_loss

        if self.verbose:
            start_time = time.time()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if self.verbose:
            end_time = time.time()
            print(f"tracking: {(end_time - start_time)*1000} ms")

        if self.enable_wandb:
            log_dict = {
                "tracking_total_loss": loss.item(),
                "tracking_sdf_loss": sdf_loss.item(),
                "tracking_color_loss": color_loss.item(),
                "tracking_depth_loss": depth_loss.item(),
            }
            # if self.use_semantic_loss:
            #     log_dict["tracking_semantic_loss"] = semantic_loss.item()
            self.wandb_run.log(log_dict)

        return loss.item()

    def update_params_from_mapping(self):
        """
        Update the parameters of scene representation from the mapping thread.
        """
        if self.mapping_idx[0] != self.prev_mapping_idx:
            # if self.verbose:
            #     print('Tracking: update the parameters from mapping')

            self.decoders.load_state_dict(self.shared_decoders.state_dict())

            for planes, self_planes in zip(
                    [self.shared_planes_xy, self.shared_planes_xz, self.shared_planes_yz],
                    [self.planes_xy, self.planes_xz, self.planes_yz]):
                for i, plane in enumerate(planes):
                    self_planes[i] = plane.detach()

            for c_planes, self_c_planes in zip(
                    [self.shared_c_planes_xy, self.shared_c_planes_xz, self.shared_c_planes_yz],
                    [self.c_planes_xy, self.c_planes_xz, self.c_planes_yz]):
                for i, c_plane in enumerate(c_planes):
                    self_c_planes[i] = c_plane.detach()

            for s_planes, self_s_planes in zip(
                    [self.shared_s_planes_xy, self.shared_s_planes_xz, self.shared_s_planes_yz],
                    [self.s_planes_xy, self.s_planes_xz, self.s_planes_yz]):
                for i, s_plane in enumerate(s_planes):
                    self_s_planes[i] = s_plane.detach()

            self.prev_mapping_idx = self.mapping_idx[0].clone()

    def run(self):
        device = self.device

        all_planes = (self.planes_xy, self.planes_xz, self.planes_yz,
                      self.c_planes_xy, self.c_planes_xz, self.c_planes_yz,
                      self.s_planes_xy, self.s_planes_xz, self.s_planes_yz)


        pbar = tqdm(self.frame_loader, smoothing=0.05)

        for idx, gt_color, gt_depth, gt_c2w, gt_semantic in pbar:
            gt_color = gt_color.to(device, non_blocking=True)
            gt_depth = gt_depth.to(device, non_blocking=True)
            gt_c2w = gt_c2w.to(device, non_blocking=True)
            gt_semantic = gt_semantic.to(device, non_blocking=True)

            if not self.verbose:
                pbar.set_description(f"Tracking Frame {idx[0]}")
            idx = idx[0]

            # initiate mapping every self.every_frame frames
            if idx > 0 and (idx % self.every_frame == 1 or self.every_frame == 1):
                while self.mapping_idx[0] != idx - 1:
                    time.sleep(0.001)
                pre_c2w = self.estimate_c2w_list[idx - 1].unsqueeze(0).to(device)

            self.update_params_from_mapping()

            if idx == 0 or self.use_gt_pose:
                c2w = gt_c2w
                if not self.no_vis_on_first_frame:
                    self.visualizer.save_imgs(idx, 0, gt_depth, gt_color, c2w.squeeze(), all_planes, self.decoders)

            else:
                if self.const_speed_assumption and idx - 2 >= 0:
                    ## Linear prediction for initialization
                    pre_poses = torch.stack([self.estimate_c2w_list[idx - 2], pre_c2w.squeeze(0)], dim=0)
                    pre_poses = matrix_to_cam_pose(pre_poses)
                    cam_pose = 2 * pre_poses[1:] - pre_poses[0:1]
                else:
                    ## Initialize with the last known pose
                    cam_pose = matrix_to_cam_pose(pre_c2w)

                T = torch.nn.Parameter(cam_pose[:, -3:].clone())
                R = torch.nn.Parameter(cam_pose[:,:4].clone())
                cam_para_list_T = [T]
                cam_para_list_R = [R]
                optimizer_camera = torch.optim.Adam([{'params': cam_para_list_T, 'lr': self.cam_lr_T, 'betas':(0.5, 0.999)},
                                                     {'params': cam_para_list_R, 'lr': self.cam_lr_R, 'betas':(0.5, 0.999)}])

                current_min_loss = torch.tensor(float('inf')).float().to(device)
                for cam_iter in range(self.num_cam_iters):
                    cam_pose = torch.cat([R, T], -1)

                    self.visualizer.save_imgs(idx, cam_iter, gt_depth, gt_color, cam_pose, all_planes, self.decoders)

                    loss = self.optimize_tracking(cam_pose, gt_color, gt_depth, self.tracking_pixels, optimizer_camera, gt_semantic)
                    if loss < current_min_loss:
                        current_min_loss = loss
                        candidate_cam_pose = cam_pose.clone().detach()

                c2w = cam_pose_to_matrix(candidate_cam_pose)

            self.estimate_c2w_list[idx] = c2w.squeeze(0).clone()
            self.gt_c2w_list[idx] = gt_c2w.squeeze(0).clone()
            pre_c2w = c2w.clone()
            self.idx[0] = idx