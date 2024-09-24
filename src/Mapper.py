# This file is a part of SNI-SLAM

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import os
import time

from src.common import (get_samples, random_select, matrix_to_cam_pose, cam_pose_to_matrix)
from src.utils.datasets import get_dataset, SeqSampler
from src.utils.Frame_Visualizer import Frame_Visualizer
from src.tools.cull_mesh import cull_mesh
import wandb

class Mapper(object):
    """
    Mapping main class.
    Args:
        cfg (dict): config dict
        args (argparse.Namespace): arguments
        sni : SNI_SLAM object
    """

    def __init__(self, cfg, args, sni):

        self.cfg = cfg
        self.args = args

        self.idx = sni.idx
        self.truncation = sni.truncation
        self.bound = sni.bound
        self.logger = sni.logger
        self.mesher = sni.mesher
        self.output = sni.output
        self.verbose = sni.verbose
        self.renderer = sni.renderer
        self.mapping_idx = sni.mapping_idx
        self.mapping_cnt = sni.mapping_cnt
        self.decoders = sni.shared_decoders

        self.planes_xy = sni.shared_planes_xy
        self.planes_xz = sni.shared_planes_xz
        self.planes_yz = sni.shared_planes_yz

        self.c_planes_xy = sni.shared_c_planes_xy
        self.c_planes_xz = sni.shared_c_planes_xz
        self.c_planes_yz = sni.shared_c_planes_yz

        self.use_gt_semantic = cfg['func']['use_gt_semantic']

        self.s_planes_xy = sni.shared_s_planes_xy
        self.s_planes_xz = sni.shared_s_planes_xz
        self.s_planes_yz = sni.shared_s_planes_yz

        self.estimate_c2w_list = sni.estimate_c2w_list
        self.mapping_first_frame = sni.mapping_first_frame

        self.model_manager = sni.model_manager

        self.enable_wandb = cfg['func']['enable_wandb']
        if self.enable_wandb:
            self.wandb_run = sni.wandb_run

        self.scale = cfg['scale']
        self.device = cfg['device']
        self.keyframe_device = cfg['keyframe_device']
        self.feature_device = cfg['feature_device']

        self.eval_rec = cfg['meshing']['eval_rec']
        self.joint_opt = False  # Even if joint_opt is enabled, it starts only when there are at least 4 keyframes
        self.joint_opt_cam_lr = cfg['mapping']['joint_opt_cam_lr'] # The learning rate for camera poses during mapping
        self.mesh_freq = cfg['mapping']['mesh_freq']
        self.ckpt_freq = cfg['mapping']['ckpt_freq']
        self.mapping_pixels = cfg['mapping']['pixels']
        self.every_frame = cfg['mapping']['every_frame']
        self.w_sdf_fs = cfg['mapping']['w_sdf_fs']
        self.w_sdf_center = cfg['mapping']['w_sdf_center']
        self.w_sdf_tail = cfg['mapping']['w_sdf_tail']
        self.w_depth = cfg['mapping']['w_depth']
        self.w_color = cfg['mapping']['w_color']

        self.w_feature = cfg['mapping']['w_feature']
        self.w_semantic = cfg['mapping']['w_semantic']

        self.keyframe_every = cfg['mapping']['keyframe_every']
        self.mapping_window_size = cfg['mapping']['mapping_window_size']
        self.no_vis_on_first_frame = cfg['mapping']['no_vis_on_first_frame']
        self.no_log_on_first_frame = cfg['mapping']['no_log_on_first_frame']
        self.no_mesh_on_first_frame = cfg['mapping']['no_mesh_on_first_frame']
        self.keyframe_selection_method = cfg['mapping']['keyframe_selection_method']

        self.c_dim = cfg['model']['c_dim']
        self.n_classes = cfg['model']['cnn']['n_classes']

        self.keyframe_dict = []
        self.keyframe_list = []
        self.frame_reader = get_dataset(cfg, args, self.scale, device=self.device)
        self.n_img = len(self.frame_reader)
        self.frame_loader = DataLoader(self.frame_reader, batch_size=1, num_workers=1, pin_memory=True,
                                       prefetch_factor=2, sampler=SeqSampler(self.n_img, self.every_frame))

        self.visualizer = Frame_Visualizer(freq=cfg['mapping']['vis_freq'], inside_freq=cfg['mapping']['vis_inside_freq'],
                                           vis_dir=os.path.join(self.output, 'mapping_vis'), renderer=self.renderer,
                                           truncation=self.truncation, verbose=self.verbose, device=self.device,
                                           n_classes=cfg['model']['cnn']['n_classes'])

        self.H, self.W, self.fx, self.fy, self.cx, self.cy = sni.H, sni.W, sni.fx, sni.fy, sni.cx, sni.cy


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

    def keyframe_selection_overlap(self, gt_color, gt_depth, c2w, num_keyframes, num_samples=8, num_rays=50):
        """
        Select overlapping keyframes to the current camera observation.

        Args:
            gt_color: ground truth color image of the current frame.
            gt_depth: ground truth depth image of the current frame.
            c2w: camera to world matrix for target view (3x4 or 4x4 both fine).
            num_keyframes (int): number of overlapping keyframes to select.
            num_samples (int, optional): number of samples/points per ray. Defaults to 8.
            num_rays (int, optional): number of pixels to sparsely sample
                from each image. Defaults to 50.
        Returns:
            selected_keyframe_list (list): list of selected keyframe id.
        """
        device = self.device
        H, W, fx, fy, cx, cy = self.H, self.W, self.fx, self.fy, self.cx, self.cy

        rays_o, rays_d, gt_depth, _, _, _, _ = get_samples(
            0, H, 0, W, num_rays, H, W, fx, fy, cx, cy,
            c2w.unsqueeze(0), gt_depth.unsqueeze(0), gt_color.unsqueeze(0), device=device, dim=self.c_dim)

        gt_depth = gt_depth.reshape(-1, 1)
        nonzero_depth = gt_depth[:, 0] > 0
        rays_o = rays_o[nonzero_depth]
        rays_d = rays_d[nonzero_depth]
        gt_depth = gt_depth[nonzero_depth]
        gt_depth = gt_depth.repeat(1, num_samples)
        t_vals = torch.linspace(0., 1., steps=num_samples).to(device)
        near = gt_depth*0.8
        far = gt_depth+0.5
        z_vals = near * (1.-t_vals) + far * (t_vals)
        pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]  # [num_rays, num_samples, 3]
        pts = pts.reshape(1, -1, 3)

        keyframes_c2ws = torch.stack([self.estimate_c2w_list[idx] for idx in self.keyframe_list], dim=0)
        w2cs = torch.inverse(keyframes_c2ws[:-2])     ## The last two keyframes are already included

        ones = torch.ones_like(pts[..., 0], device=device).reshape(1, -1, 1)
        homo_pts = torch.cat([pts, ones], dim=-1).reshape(1, -1, 4, 1).expand(w2cs.shape[0], -1, -1, -1)
        w2cs_exp = w2cs.unsqueeze(1).expand(-1, homo_pts.shape[1], -1, -1)
        cam_cords_homo = w2cs_exp @ homo_pts
        cam_cords = cam_cords_homo[:, :, :3]
        K = torch.tensor([[fx, .0, cx], [.0, fy, cy],
                          [.0, .0, 1.0]], device=device).reshape(3, 3)
        cam_cords[:, :, 0] *= -1
        uv = K @ cam_cords
        z = uv[:, :, -1:] + 1e-5
        uv = uv[:, :, :2] / z
        edge = 20
        mask = (uv[:, :, 0] < W - edge) * (uv[:, :, 0] > edge) * \
               (uv[:, :, 1] < H - edge) * (uv[:, :, 1] > edge)
        mask = mask & (z[:, :, 0] < 0)
        mask = mask.squeeze(-1)
        percent_inside = mask.sum(dim=1) / uv.shape[1]

        ## Considering only overlapped frames
        selected_keyframes = torch.nonzero(percent_inside).squeeze(-1)
        rnd_inds = torch.randperm(selected_keyframes.shape[0])
        selected_keyframes = selected_keyframes[rnd_inds[:num_keyframes]]

        selected_keyframes = list(selected_keyframes.cpu().numpy())

        return selected_keyframes

    # def get_feature_from_ray(self, batch_rays_o, batch_rays_d, kf_sem_feats, kf_rgb_feats):

    def optimize_mapping(self, iters, lr_factor, idx, cur_gt_color, cur_gt_depth, gt_cur_c2w, cur_sem_feat,
                         cur_sem_label, gt_label, keyframe_dict, keyframe_list, cur_c2w):
        all_planes = (self.planes_xy, self.planes_xz, self.planes_yz,
                      self.c_planes_xy, self.c_planes_xz, self.c_planes_yz,
                      self.s_planes_xy, self.s_planes_xz, self.s_planes_yz)
        H, W, fx, fy, cx, cy = self.H, self.W, self.fx, self.fy, self.cx, self.cy
        cfg = self.cfg
        device = self.device

        if len(keyframe_dict) == 0:
            optimize_frame = []
        else:
            if self.keyframe_selection_method == 'global':
                optimize_frame = random_select(len(self.keyframe_dict)-2, self.mapping_window_size-1)
            elif self.keyframe_selection_method == 'overlap':
                optimize_frame = self.keyframe_selection_overlap(cur_gt_color, cur_gt_depth, cur_c2w, self.mapping_window_size-1)

        # add the last two keyframes and the current frame(use -1 to denote)
        if len(keyframe_list) > 1:
            optimize_frame = optimize_frame + [len(keyframe_list)-1] + [len(keyframe_list)-2]
            optimize_frame = sorted(optimize_frame)
        optimize_frame += [-1]  ## -1 represents the current frame

        pixs_per_image = self.mapping_pixels//len(optimize_frame)

        decoders_para_list = []
        decoders_para_list += list(self.decoders.parameters())

        planes_para = []
        for planes in [self.planes_xy, self.planes_xz, self.planes_yz]:
            for i, plane in enumerate(planes):
                plane = nn.Parameter(plane)
                planes_para.append(plane)
                planes[i] = plane

        c_planes_para = []
        for c_planes in [self.c_planes_xy, self.c_planes_xz, self.c_planes_yz]:
            for i, c_plane in enumerate(c_planes):
                c_plane = nn.Parameter(c_plane)
                c_planes_para.append(c_plane)
                c_planes[i] = c_plane

        s_planes_para = []
        for s_planes in [self.s_planes_xy, self.s_planes_xz, self.s_planes_yz]:
            for i, s_plane in enumerate(s_planes):
                s_plane = nn.Parameter(s_plane)
                s_planes_para.append(s_plane)
                s_planes[i] = s_plane

        gt_depths = []
        gt_colors = []
        c2ws = []
        gt_c2ws = []
        for frame in optimize_frame:
            if frame != -1:
                gt_depths.append(keyframe_dict[frame]['depth'].to(device))
                gt_colors.append(keyframe_dict[frame]['color'].to(device))
                c2ws.append(keyframe_dict[frame]['est_c2w'])
                gt_c2ws.append(keyframe_dict[frame]['gt_c2w'])

            else:
                gt_depths.append(cur_gt_depth)
                gt_colors.append(cur_gt_color)
                c2ws.append(cur_c2w)
                gt_c2ws.append(gt_cur_c2w)

        gt_depths = torch.stack(gt_depths, dim=0)
        gt_colors = torch.stack(gt_colors, dim=0)
        c2ws = torch.stack(c2ws, dim=0)

        kf_sem_feats = []
        kf_rgb_feats = []
        kf_gt_label = []
        for frame in optimize_frame:
            if frame != -1:
                sem_feat = keyframe_dict[frame]['sem_feat'].to(device)
                rgb_feat = self.model_manager.head(sem_feat)
                kf_sem_feats.append(sem_feat.squeeze(0))
                kf_rgb_feats.append(rgb_feat.squeeze(0))

                gt_sem_label = keyframe_dict[frame]['gt_sem_label'].to(device)
                kf_gt_label.append(gt_sem_label)

            else:
                rgb_feat = self.model_manager.head(cur_sem_feat)
                kf_sem_feats.append(cur_sem_feat.squeeze(0))
                kf_rgb_feats.append(rgb_feat.squeeze(0))
                kf_gt_label.append(cur_sem_label)

        kf_sem_feats = torch.stack(kf_sem_feats, dim=0)
        kf_rgb_feats = torch.stack(kf_rgb_feats, dim=0)
        kf_gt_label = torch.stack(kf_gt_label, dim=0)

        if self.joint_opt:
            cam_poses = nn.Parameter(matrix_to_cam_pose(c2ws[1:]))

            model_paras = [{'params': decoders_para_list, 'lr': 0},
                           {'params': planes_para, 'lr': 0},
                           {'params': c_planes_para, 'lr': 0},
                           {'params': [cam_poses], 'lr': 0}]

        else:
            model_paras = [{'params': decoders_para_list, 'lr': 0},
                           {'params': planes_para, 'lr': 0},
                           {'params': c_planes_para, 'lr': 0}]

        model_paras.append({'params': s_planes_para, 'lr': 5e-3})
        model_paras.append({'params': self.model_manager.encoder.parameters(), 'lr': 3e-3})
        model_paras.append({'params': self.model_manager.head.parameters(), 'lr': 3e-3})
        model_paras.append({'params': self.model_manager.feat_fusion.parameters(), 'lr': cfg['mapping']['lr']['fusion_lr']})

        optimizer = torch.optim.Adam(model_paras)
        optimizer.param_groups[0]['lr'] = cfg['mapping']['lr']['decoders_lr'] * lr_factor
        optimizer.param_groups[1]['lr'] = cfg['mapping']['lr']['planes_lr'] * lr_factor
        optimizer.param_groups[2]['lr'] = cfg['mapping']['lr']['c_planes_lr'] * lr_factor

        if self.joint_opt:
            optimizer.param_groups[3]['lr'] = self.joint_opt_cam_lr

        for joint_iter in range(iters):
            if self.verbose:
                start_time = time.time()

            if (not (idx == 0 and self.no_vis_on_first_frame)):
                self.visualizer.save_imgs(idx, joint_iter, cur_gt_depth, cur_gt_color, cur_c2w, all_planes, self.decoders,
                                          gt_sem=gt_label, est_sem=cur_sem_label)

            if self.joint_opt:
                ## We fix the oldest c2w to avoid drifting
                c2ws_ = torch.cat([c2ws[0:1], cam_pose_to_matrix(cam_poses)], dim=0)
            else:
                c2ws_ = c2ws

            batch_rays_o, batch_rays_d, batch_gt_depth, batch_gt_color, batch_sem_feats, batch_rgb_feats, batch_gt_label = get_samples(
                0, H, 0, W, pixs_per_image, H, W, fx, fy, cx, cy, c2ws_, gt_depths, gt_colors,
                sem_feats=kf_sem_feats, rgb_feats=kf_rgb_feats, gt_label=kf_gt_label, device=device, dim=self.c_dim)

            depth, color, sdf, z_vals, gt_feat, plane_feat, render_semantic = self.renderer.render_batch_ray(all_planes, self.decoders,
                                                            batch_rays_d, batch_rays_o, device, self.truncation, gt_depth=batch_gt_depth,
                                                            sem_feats=batch_sem_feats, rgb_feats=batch_rgb_feats,
                                                            return_emb=True)
            depth_mask = (batch_gt_depth > 0)

            ## SDF losses
            sdf_loss = self.sdf_losses(sdf[depth_mask], z_vals[depth_mask], batch_gt_depth[depth_mask])
            loss = sdf_loss

            ## Color loss
            color_loss = self.w_color * torch.square(batch_gt_color - color).mean()
            loss = loss + color_loss

            ### Depth loss
            depth_loss = self.w_depth * torch.square(batch_gt_depth[depth_mask] - depth[depth_mask]).mean()
            loss = loss + depth_loss

            ### feature loss
            plane_feature = plane_feat.detach()
            gt_feature = gt_feat.detach()
            feature_loss = self.w_feature * (gt_feature - plane_feature).abs().mean()
            loss = loss + feature_loss

            CrossEntropyLoss = nn.CrossEntropyLoss(ignore_index=-1)
            semantic_loss = self.w_semantic * CrossEntropyLoss(render_semantic, batch_gt_label)
            loss = loss + semantic_loss

            optimizer.zero_grad()
            loss.backward(retain_graph=False)
            optimizer.step()

            if self.verbose:
                end_time = time.time()
                print(f"mapping: {(end_time - start_time)*1000} ms")

            if self.enable_wandb:
                log_dict = {
                    "total_loss": loss.item(),
                    "sdf_loss": sdf_loss.item(),
                    "color_loss": color_loss.item(),
                    "depth_loss": depth_loss.item(),
                }

                log_dict["feature_loss"] = feature_loss.item()
                log_dict["semantic_loss"] = semantic_loss.item()
                self.wandb_run.log(log_dict)

        if self.joint_opt:
            # put the updated camera poses back
            optimized_c2ws = cam_pose_to_matrix(cam_poses.detach())

            camera_tensor_id = 0
            for frame in optimize_frame[1:]:
                if frame != -1:
                    keyframe_dict[frame]['est_c2w'] = optimized_c2ws[camera_tensor_id]
                    camera_tensor_id += 1
                else:
                    cur_c2w = optimized_c2ws[-1]

        return cur_c2w

    def add_noise(self, gt_sem_label, noise_ratio):
        h,w = gt_sem_label.shape
        noise_label = gt_sem_label.clone()
        import random
        num_classes = self.n_classes
        for i in range(h):
            for j in range(w):
                if random.random() < noise_ratio:
                    noise_label[i, j] = random.randint(0, num_classes - 1)
        return noise_label

    def run(self):
        cfg = self.cfg

        all_planes = (
            self.planes_xy, self.planes_xz, self.planes_yz,
            self.c_planes_xy, self.c_planes_xz, self.c_planes_yz,
            self.s_planes_xy, self.s_planes_xz, self.s_planes_yz)

        idx, gt_color, gt_depth, gt_c2w, gt_semantic = self.frame_reader[0]
        data_iterator = iter(self.frame_loader)

        ## Fixing the first camera pose
        self.estimate_c2w_list[0] = gt_c2w

        init_phase = True
        prev_idx = -1
        while True:
            while True:
                idx = self.idx[0].clone()
                if idx == self.n_img-1: ## Last input frame
                    break

                if idx % self.every_frame == 0 and idx != prev_idx:
                    break

                time.sleep(0.001)

            prev_idx = idx

            _, gt_color, gt_depth, gt_c2w, gt_semantic = next(data_iterator)
            gt_color = gt_color.squeeze(0).to(self.device, non_blocking=True)
            gt_depth = gt_depth.squeeze(0).to(self.device, non_blocking=True)
            gt_c2w = gt_c2w.squeeze(0).to(self.device, non_blocking=True)
            gt_semantic = gt_semantic.squeeze(0).to(self.device)

            cur_c2w = self.estimate_c2w_list[idx]

            if not init_phase:
                lr_factor = cfg['mapping']['lr_factor']
                iters = cfg['mapping']['iters']
            else:
                lr_factor = cfg['mapping']['lr_first_factor']
                iters = cfg['mapping']['iters_first']

            ## Deciding if camera poses should be jointly optimized
            self.joint_opt = (len(self.keyframe_list) > 4) and cfg['mapping']['joint_opt']

            with torch.no_grad():
                frame_rgb = gt_color.permute(2, 0, 1).unsqueeze(0).to(self.device)  # net_rgb:[1,16,680,1200]
                self.model_manager.set_mode_feature()
                sem_feat = self.model_manager.cnn(frame_rgb)

                if self.use_gt_semantic:
                    gt_sem_label = gt_semantic

                else:
                    self.model_manager.set_mode_result()
                    gt_sem_label = self.model_manager.cnn(frame_rgb)

            cur_c2w = self.optimize_mapping(iters, lr_factor, idx, gt_color, gt_depth, gt_c2w, sem_feat,
                                            gt_sem_label, gt_semantic,
                                            self.keyframe_dict, self.keyframe_list, cur_c2w)

            if self.joint_opt:
                self.estimate_c2w_list[idx] = cur_c2w

            # add new frame to keyframe set
            if idx % self.keyframe_every == 0:
                self.keyframe_list.append(idx)

                frame_dict = {
                    'gt_c2w': gt_c2w,
                    'idx': idx,
                    'color': gt_color.to(self.keyframe_device),
                    'depth': gt_depth.to(self.keyframe_device),
                    'est_c2w': cur_c2w.clone(),
                }
                frame_dict['sem_feat'] = sem_feat.to(self.feature_device)
                frame_dict['gt_sem_label'] = gt_sem_label.to(self.feature_device)

                self.keyframe_dict.append(frame_dict)

            init_phase = False
            self.mapping_first_frame[0] = 1     # mapping of first frame is done, can begin tracking

            if ((not (idx == 0 and self.no_log_on_first_frame)) and idx % self.ckpt_freq == 0) or idx == self.n_img-1:
                self.logger.log(idx, self.keyframe_list)

            self.mapping_idx[0] = idx
            self.mapping_cnt[0] += 1

            if (idx % self.mesh_freq == 0) and (not (idx == 0 and self.no_mesh_on_first_frame)):
                # mesh_out_file = f'{self.output}/mesh/{idx:05d}_mesh_sem.ply'
                # self.mesher.get_mesh(mesh_out_file, all_planes, self.decoders, self.keyframe_dict, self.device)
                # cull_mesh(mesh_out_file, self.cfg, self.args, self.device, estimate_c2w_list=self.estimate_c2w_list[:idx+1])
                mesh_out_semantic = f'{self.output}/mesh/{idx:05d}_mesh_sem.ply'
                mesh_out_color = f'{self.output}/mesh/{idx:05d}_mesh_rgb.ply'
                self.mesher.get_mesh(mesh_out_color, all_planes, self.decoders, self.keyframe_dict, self.device, mesh_out_semantic=mesh_out_semantic, color=False)

            if idx == self.n_img-1:
                mesh_out_semantic = f'{self.output}/mesh/final_mesh_semantic.ply'
                mesh_out_color = f'{self.output}/mesh/final_mesh_color.ply'
                self.mesher.get_mesh(mesh_out_color, all_planes, self.decoders, self.keyframe_dict, self.device, mesh_out_semantic=mesh_out_semantic, semantic=False)
                cull_mesh(mesh_out_color, self.cfg, self.args, self.device, estimate_c2w_list=self.estimate_c2w_list)

                break

            if idx == self.n_img-1:
                break