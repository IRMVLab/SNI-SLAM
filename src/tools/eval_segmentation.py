# This file is a part of SNI-SLAM.
import sys
sys.path.append('.')

import torch, argparse, os
from src.common import get_rays, sample_pdf, normalize_3d_coordinate
from src import config
import numpy as np
from src.utils.datasets import get_dataset
from src.tools.segmentationMetric import SegmentationMetric
import matplotlib.pyplot as plt

all_pixAccs = []
all_mIoUs = []

class Eval_Segmentation():
    def __init__(self, cfg, device='cuda'):
        self.device = device
        self.H, self.W, self.fx, self.fy, self.cx, self.cy = cfg['cam']['H'], cfg['cam'][
            'W'], cfg['cam']['fx'], cfg['cam']['fy'], cfg['cam']['cx'], cfg['cam']['cy']
        self.ray_batch_size = 10000

        self.perturb = cfg['rendering']['perturb']
        self.n_stratified = cfg['rendering']['n_stratified']
        self.n_importance = cfg['rendering']['n_importance']
        self.scale = cfg['scale']
        self.load_bound(cfg)
        self.n_classes = cfg['model']['cnn']['n_classes']

        self.metric = SegmentationMetric(self.n_classes)
        self.vis_dir = cfg['data']['output'] + '/vis'
        os.makedirs(self.vis_dir, exist_ok=True)

    def load_bound(self, cfg):
        self.bound = torch.from_numpy(np.array(cfg['mapping']['bound'])*self.scale).float().to(self.device)
        bound_dividable = cfg['planes_res']['bound_dividable']
        self.bound[:, 1] = (((self.bound[:, 1]-self.bound[:, 0]) /
                            bound_dividable).int()+1)*bound_dividable+self.bound[:, 0]

    def sdf2alpha(self, sdf, beta=10):
        return 1. - torch.exp(-beta * torch.sigmoid(-sdf * beta))

    def render_img(self, all_planes, decoders, c2w, truncation, device, gt_depth=None):
        with torch.no_grad():
            H = self.H
            W = self.W
            rays_o, rays_d = get_rays(H, W, self.fx, self.fy, self.cx, self.cy, c2w, device)
            rays_o = rays_o.reshape(-1, 3)
            rays_d = rays_d.reshape(-1, 3)

            semantic_list = []
            rgb_list = []

            ray_batch_size = self.ray_batch_size
            gt_depth = gt_depth.reshape(-1)

            for i in range(0, rays_d.shape[0], ray_batch_size):
                rays_d_batch = rays_d[i:i + ray_batch_size]
                rays_o_batch = rays_o[i:i + ray_batch_size]
                gt_depth_batch = gt_depth[i:i + ray_batch_size]
                semantic, rgb = self.render_batch_ray(all_planes, decoders, rays_d_batch, rays_o_batch,
                                            device, truncation, gt_depth=gt_depth_batch)
                semantic_list.append(semantic)
                rgb_list.append(rgb)

            semantic = torch.cat(semantic_list, dim=0)
            semantic = semantic.reshape(H, W, -1)

            rgb = torch.cat(rgb_list, dim=0)
            rgb = rgb.reshape(H, W, -1)

            return semantic, rgb

    def visualize(self, idx, semantic, gt_semantic, rgb, gt_rgb):
        pred = semantic.detach().permute(2, 0, 1).unsqueeze(0)
        gt = gt_semantic.squeeze().unsqueeze(0).unsqueeze(0)

        semantic_gt = torch.zeros_like(gt, dtype=torch.float32)
        semantic_gt = torch.tile(semantic_gt, (1, self.n_classes, 1, 1))
        for channel in range(self.n_classes):
            channel1 = channel * 1.0
            semantic_gt[0, channel, :, :] = gt * (gt[0, 0, :, :] == channel1)

        self.metric.update(pred, semantic_gt)
        pixAcc, mIoU = self.metric.get()
        all_pixAccs.append(pixAcc)
        all_mIoUs.append(mIoU)

        semantic = torch.max(semantic, 2).indices.squeeze()
        pred_label_image = self.decode_segmap(image=semantic.detach().cpu(), nc=self.n_classes)
        gt_label_image = self.decode_segmap(image=gt_semantic.cpu(), nc=self.n_classes)

        psnr = self.calc_psnr(rgb.cpu(), gt_rgb.cpu()).mean()
        print(f"psnr: {psnr}")

        fig, axs = plt.subplots(2, 2)
        fig.tight_layout()

        axs[0, 0].imshow(pred_label_image, cmap="plasma")
        axs[0, 0].set_title('Predicted Label')
        axs[0, 0].set_xticks([])
        axs[0, 0].set_yticks([])
        axs[0, 1].imshow(gt_label_image, cmap="plasma")
        axs[0, 1].set_title('Ground Truth Label')
        axs[0, 1].set_xticks([])
        axs[0, 1].set_yticks([])
        axs[0, 1].axis('off')

        pred_image = rgb.cpu().numpy()
        gt_image = gt_rgb.cpu().numpy()
        axs[1, 0].imshow(pred_image, cmap="plasma")
        axs[1, 0].set_title('Predicted Label')
        axs[1, 0].set_xticks([])
        axs[1, 0].set_yticks([])
        axs[1, 1].imshow(gt_image, cmap="plasma")
        axs[1, 1].set_title('Ground Truth Label')
        axs[1, 1].set_xticks([])
        axs[1, 1].set_yticks([])
        axs[1, 1].axis('off')

        plt.subplots_adjust(wspace=0, hspace=0)
        plt.savefig(f'{self.vis_dir}/{idx:05d}.jpg', bbox_inches='tight', pad_inches=0.2, dpi=300)
        plt.cla()
        plt.clf()

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

    def render_batch_ray(self, all_planes, decoders, rays_d, rays_o, device, truncation, gt_depth=None):
        n_stratified = self.n_stratified
        n_importance = self.n_importance
        n_rays = rays_o.shape[0]

        z_vals = torch.empty([n_rays, n_stratified + n_importance], device=device)  # [10000, 40]
        near = 0.0
        t_vals_uni = torch.linspace(0., 1., steps=n_stratified, device=device)
        t_vals_surface = torch.linspace(0., 1., steps=n_importance, device=device)

        ### pixels with gt depth:
        gt_depth = gt_depth.reshape(-1, 1)
        gt_mask = (gt_depth > 0).squeeze()
        gt_nonezero = gt_depth[gt_mask]

        ## Sampling points around the gt depth (surface)
        gt_depth_surface = gt_nonezero.expand(-1, n_importance)
        z_vals_surface = gt_depth_surface - (1.5 * truncation) + (3 * truncation * t_vals_surface)

        gt_depth_free = gt_nonezero.expand(-1, n_stratified)
        z_vals_free = near + 1.2 * gt_depth_free * t_vals_uni

        z_vals_nonzero, _ = torch.sort(torch.cat([z_vals_free, z_vals_surface], dim=-1), dim=-1)
        if self.perturb:
            z_vals_nonzero = self.perturbation(z_vals_nonzero)
        z_vals[gt_mask] = z_vals_nonzero

        ### pixels without gt depth (importance sampling):
        if not gt_mask.all():
            with torch.no_grad():
                rays_o_uni = rays_o[~gt_mask].detach()
                rays_d_uni = rays_d[~gt_mask].detach()
                det_rays_o = rays_o_uni.unsqueeze(-1)  # (N, 3, 1)
                det_rays_d = rays_d_uni.unsqueeze(-1)  # (N, 3, 1)
                t = (self.bound.unsqueeze(0) - det_rays_o)/det_rays_d  # (N, 3, 2)
                far_bb, _ = torch.min(torch.max(t, dim=2)[0], dim=1)
                far_bb = far_bb.unsqueeze(-1)
                far_bb += 0.01

                z_vals_uni = near * (1. - t_vals_uni) + far_bb * t_vals_uni
                if self.perturb:
                    z_vals_uni = self.perturbation(z_vals_uni)
                pts_uni = rays_o_uni.unsqueeze(1) + rays_d_uni.unsqueeze(1) * z_vals_uni.unsqueeze(-1)  # [n_rays, n_stratified, 3]

                pts_uni_nor = normalize_3d_coordinate(pts_uni.clone(), self.bound)
                sdf_uni, _ = decoders.get_raw_sdf(pts_uni_nor, all_planes[:6])
                sdf_uni = sdf_uni.reshape(*pts_uni.shape[0:2])
                alpha_uni = self.sdf2alpha(sdf_uni, decoders.beta)
                weights_uni = alpha_uni * torch.cumprod(torch.cat([torch.ones((alpha_uni.shape[0], 1), device=device)
                                                        , (1. - alpha_uni + 1e-10)], -1), -1)[:, :-1]

                z_vals_uni_mid = .5 * (z_vals_uni[..., 1:] + z_vals_uni[..., :-1])
                z_samples_uni = sample_pdf(z_vals_uni_mid, weights_uni[..., 1:-1], n_importance, det=False, device=device)
                z_vals_uni, ind = torch.sort(torch.cat([z_vals_uni, z_samples_uni], -1), -1)
                z_vals[~gt_mask] = z_vals_uni

        pts = rays_o[..., None, :] + rays_d[..., None, :] * \
              z_vals[..., :, None]  # [n_rays, n_stratified+n_importance, 3]

        raw, plane_feat = decoders(pts, all_planes)

        semantic_alpha = self.sdf2alpha(raw[..., 3], decoders.semantic_beta)
        semantic_weights = semantic_alpha * torch.cumprod(
            torch.cat([torch.ones((semantic_alpha.shape[0], 1), device=device)
                          , (1. - semantic_alpha + 1e-10)], -1), -1)[:, :-1]
        rendered_semantic = torch.sum(semantic_weights[..., None] * raw[..., 4:], -2)

        alpha_rgb = self.sdf2alpha(raw[..., 3], decoders.beta)
        weights_rgb = alpha_rgb * torch.cumprod(torch.cat([torch.ones((alpha_rgb.shape[0], 1), device=device)
                                                ,(1. - alpha_rgb + 1e-10)], -1), -1)[:, :-1]
        rendered_rgb = torch.sum(weights_rgb[..., None] * raw[..., :3], -2)

        return rendered_semantic, rendered_rgb

    def perturbation(self, z_vals):
        mids = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
        upper = torch.cat([mids, z_vals[..., -1:]], -1)
        lower = torch.cat([z_vals[..., :1], mids], -1)
        t_rand = torch.rand(z_vals.shape, device=z_vals.device)

        return lower + (upper - lower) * t_rand

    def calc_psnr(self, img1, img2):
        mse = ((img1 - img2) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
        return 20 * torch.log10(1.0 / torch.sqrt(mse))


def load_poses(pose_path):
    poses = []
    with open(pose_path, "r") as f:
        lines = f.readlines()
        num_poses = 2000   #TODO: 对应一共多少帧pose
    for i in range(num_poses):
        line = lines[i]
        c2w = np.array(list(map(float, line.split()))).reshape(4, 4)
        c2w[:3, 1] *= -1
        c2w[:3, 2] *= -1
        c2w = torch.from_numpy(c2w).float()
        poses.append(c2w)
    return poses

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Arguments to evaluate the reconstruction.'
    )
    parser.add_argument('config', type=str, help='Path to config file.')
    args = parser.parse_args()
    args.input_folder = None

    cfg = config.load_config(args.config, 'configs/SNI-SLAM.yaml')
    output = cfg['data']['output']
    ckptsdir = f'{output}/ckpts'

    eval = Eval_Segmentation(cfg)
    truncation = cfg['model']['truncation']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    frame_reader = get_dataset(cfg, args, 1, device=device)

    pose_path = os.path.join(cfg['data']['input_folder'], "traj_w_c.txt")
    c2w_list = load_poses(pose_path)

    if os.path.exists(ckptsdir):
        ckpts = [os.path.join(ckptsdir, f)
                 for f in sorted(os.listdir(ckptsdir)) if 'tar' in f]
        if len(ckpts) > 0:
            ckpt_path = ckpts[-1]
            print('Get ckpt :', ckpt_path)
            ckpt = torch.load(ckpt_path, map_location=device)

            model = config.get_model(cfg)
            model.load_state_dict(ckpt['decoder_state_dict'])
            model.bound = eval.bound
            model.to(device)

            # c2w_list = ckpt['estimate_c2w_list']

            s_planes_xy = ckpt['s_planes_xy']
            s_planes_xz = ckpt['s_planes_xz']
            s_planes_yz = ckpt['s_planes_yz']

            planes_xy = ckpt['planes_xy']
            planes_xz = ckpt['planes_xz']
            planes_yz = ckpt['planes_yz']

            c_planes_xy = ckpt['c_planes_xy']
            c_planes_xz = ckpt['c_planes_xz']
            c_planes_yz = ckpt['c_planes_yz']

            all_planes = (planes_xy, planes_xz, planes_yz, c_planes_xy, c_planes_xz, c_planes_yz,
                            s_planes_xy, s_planes_xz, s_planes_yz)
            # print(len(c2w_list))
            # for i in range(0, 200, 5):
            id = 0
            _, gt_color, gt_depth, _, semantic_data = frame_reader[id]
            # gt_semantic = torch.from_numpy(semantic_data)
            gt_semantic = semantic_data.to(device)
            gt_depth = gt_depth.to(device)

            semantic, rgb = eval.render_img(all_planes, model, c2w_list[id].to(device), truncation, device, gt_depth=gt_depth)
            eval.visualize(id, semantic, gt_semantic, rgb, gt_color)

            avg_pixAcc = sum(all_pixAccs) / len(all_pixAccs)
            avg_mIoU = sum(all_mIoUs) / len(all_mIoUs)

            print(f"Average pixel accuracy: {avg_pixAcc}")
            print(f"Average mIoU: {avg_mIoU}")