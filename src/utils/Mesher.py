# This file is a part of SNI-SLAM.
#
# This file is a modified version of https://github.com/idiap/ESLAM/blob/main/src/utils/Mesher.py
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

import numpy as np
import open3d as o3d
import skimage
import torch
import trimesh
from packaging import version
from src.utils.datasets import get_dataset

class Mesher(object):
    def __init__(self, cfg, args, sni, points_batch_size=500000, ray_batch_size=100000):
        self.points_batch_size = points_batch_size
        self.ray_batch_size = ray_batch_size
        self.renderer = sni.renderer
        self.scale = cfg['scale']

        self.resolution = cfg['meshing']['resolution']
        self.level_set = cfg['meshing']['level_set']
        self.mesh_bound_scale = cfg['meshing']['mesh_bound_scale']

        self.bound = sni.bound
        self.verbose = sni.verbose

        self.marching_cubes_bound = torch.from_numpy(
            np.array(cfg['mapping']['marching_cubes_bound']) * self.scale)

        self.frame_reader = get_dataset(cfg, args, self.scale, device='cpu')
        self.n_img = len(self.frame_reader)

        self.n_classes = cfg['model']['cnn']['n_classes']

        self.H, self.W, self.fx, self.fy, self.cx, self.cy = sni.H, sni.W, sni.fx, sni.fy, sni.cx, sni.cy

    def get_bound_from_frames(self, keyframe_dict, scale=1):
        """
        Get the scene bound (convex hull),
        using sparse estimated camera poses and corresponding depth images.

        Args:
            keyframe_dict (list): list of keyframe info dictionary.
            scale (float): scene scale.

        Returns:
            return_mesh (trimesh.Trimesh): the convex hull.
        """

        H, W, fx, fy, cx, cy = self.H, self.W, self.fx, self.fy, self.cx, self.cy

        if version.parse(o3d.__version__) >= version.parse('0.13.0'):
            # for new version as provided in environment.yaml
            volume = o3d.pipelines.integration.ScalableTSDFVolume(
                voxel_length=4.0 * scale / 512.0,
                sdf_trunc=0.04 * scale,
                color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
        else:
            # for lower version
            volume = o3d.integration.ScalableTSDFVolume(
                voxel_length=4.0 * scale / 512.0,
                sdf_trunc=0.04 * scale,
                color_type=o3d.integration.TSDFVolumeColorType.RGB8)
        cam_points = []
        for keyframe in keyframe_dict:
            c2w = keyframe['est_c2w'].cpu().numpy()
            # convert to open3d camera pose
            c2w[:3, 1] *= -1.0
            c2w[:3, 2] *= -1.0
            w2c = np.linalg.inv(c2w)
            cam_points.append(c2w[:3, 3])
            depth = keyframe['depth'].cpu().numpy()
            color = keyframe['color'].cpu().numpy()

            depth = o3d.geometry.Image(depth.astype(np.float32))
            color = o3d.geometry.Image(np.array(
                (color * 255).astype(np.uint8)))

            intrinsic = o3d.camera.PinholeCameraIntrinsic(W, H, fx, fy, cx, cy)
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                color,
                depth,
                depth_scale=1,
                depth_trunc=1000,
                convert_rgb_to_intensity=False)
            volume.integrate(rgbd, intrinsic, w2c)

        cam_points = np.stack(cam_points, axis=0)
        mesh = volume.extract_triangle_mesh()
        mesh_points = np.array(mesh.vertices)
        points = np.concatenate([cam_points, mesh_points], axis=0)
        o3d_pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
        mesh, _ = o3d_pc.compute_convex_hull()
        mesh.compute_vertex_normals()
        if version.parse(o3d.__version__) >= version.parse('0.13.0'):
            mesh = mesh.scale(self.mesh_bound_scale, mesh.get_center())
        else:
            mesh = mesh.scale(self.mesh_bound_scale, center=True)
        points = np.array(mesh.vertices)
        faces = np.array(mesh.triangles)
        return_mesh = trimesh.Trimesh(vertices=points, faces=faces)
        return return_mesh

    def eval_points(self, p, all_planes, decoders, color=False, semantic=False):
        """
        Evaluates the TSDF and/or color value for the points.
        Args:
            p (torch.Tensor): points to be evaluated, shape (N, 3).
            all_planes (Tuple): all feature planes.
            decoders (torch.nn.Module): decoders for TSDF and color.
        Returns:
            ret (torch.Tensor): the evaluation result, shape (N, 4).
        """

        p_split = torch.split(p, self.points_batch_size)
        bound = self.bound
        rets_color = []
        rets_semantic = []
        for pi in p_split:
            # mask for points out of bound
            mask_x = (pi[:, 0] < bound[0][1]) & (pi[:, 0] > bound[0][0])
            mask_y = (pi[:, 1] < bound[1][1]) & (pi[:, 1] > bound[1][0])
            mask_z = (pi[:, 2] < bound[2][1]) & (pi[:, 2] > bound[2][0])
            mask = mask_x & mask_y & mask_z

            ret, _ = decoders(pi, all_planes=all_planes)    # [N,3+1+num_classes]

            ret[~mask, 3] = -1
            if color:
                rets_color.append(ret[:, :4])
            if semantic:
                semantic_val = torch.max(ret[:, 4:], dim=1).indices.squeeze()
                img_semantic = self.decode_segmap(image=semantic_val.detach().cpu(), nc=self.n_classes)
                rets_semantic.append(img_semantic)

        if color:
            ret = torch.cat(rets_color, dim=0)
        if semantic:
            ret = torch.cat(rets_semantic, dim=0)

        return ret

    def get_grid_uniform(self, resolution):
        """
        Get query point coordinates for marching cubes.

        Args:
            resolution (int): marching cubes resolution.

        Returns:
            (dict): points coordinates and sampled coordinates for each axis.
        """
        bound = self.marching_cubes_bound

        padding = 0.05

        nsteps_x = ((bound[0][1] - bound[0][0] + 2 * padding) / resolution).round().int().item()
        x = np.linspace(bound[0][0] - padding, bound[0][1] + padding, nsteps_x)
        
        nsteps_y = ((bound[1][1] - bound[1][0] + 2 * padding) / resolution).round().int().item()
        y = np.linspace(bound[1][0] - padding, bound[1][1] + padding, nsteps_y)
        
        nsteps_z = ((bound[2][1] - bound[2][0] + 2 * padding) / resolution).round().int().item()
        z = np.linspace(bound[2][0] - padding, bound[2][1] + padding, nsteps_z)

        x_t, y_t, z_t = torch.from_numpy(x).float(), torch.from_numpy(y).float(), torch.from_numpy(z).float()
        grid_x, grid_y, grid_z = torch.meshgrid(x_t, y_t, z_t, indexing='xy')
        grid_points_t = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1), grid_z.reshape(-1)], dim=1)

        return {"grid_points": grid_points_t, "xyz": [x, y, z]}

    def get_mesh(self, mesh_out_color, all_planes, decoders, keyframe_dict, device='cuda:0', mesh_out_semantic=None, color=True, semantic=True):
        """
        Get mesh from keyframes and feature planes and save to file.
        Args:
            mesh_out_file (str): output mesh file.
            all_planes (Tuple): all feature planes.
            decoders (torch.nn.Module): decoders for TSDF and color.
            keyframe_dict (dict): keyframe dictionary.
            device (str): device to run the model.
            color (bool): whether to use color.
        Returns:
            None

        """

        with torch.no_grad():
            grid = self.get_grid_uniform(self.resolution)
            points = grid['grid_points']
            mesh_bound = self.get_bound_from_frames(keyframe_dict, self.scale)

            z = []
            mask = []
            for i, pnts in enumerate(torch.split(points, self.points_batch_size, dim=0)):
                mask.append(mesh_bound.contains(pnts.cpu().numpy()))
            mask = np.concatenate(mask, axis=0)

            for i, pnts in enumerate(torch.split(points, self.points_batch_size, dim=0)):
                z.append(self.eval_points(pnts.to(device), all_planes, decoders, color=True).cpu().numpy()[:, 3])
            z = np.concatenate(z, axis=0)
            z[~mask] = -1

            try:
                if version.parse(
                        skimage.__version__) > version.parse('0.15.0'):
                    # for new version as provided in environment.yaml
                    verts, faces, normals, values = skimage.measure.marching_cubes(
                        volume=z.reshape(
                            grid['xyz'][1].shape[0], grid['xyz'][0].shape[0],
                            grid['xyz'][2].shape[0]).transpose([1, 0, 2]),
                        level=self.level_set,
                        spacing=(grid['xyz'][0][2] - grid['xyz'][0][1],
                                 grid['xyz'][1][2] - grid['xyz'][1][1],
                                 grid['xyz'][2][2] - grid['xyz'][2][1]))
                else:
                    # for lower version
                    print('lower version')
                    verts, faces, normals, values = skimage.measure.marching_cubes_lewiner(
                        volume=z.reshape(
                            grid['xyz'][1].shape[0], grid['xyz'][0].shape[0],
                            grid['xyz'][2].shape[0]).transpose([1, 0, 2]),
                        level=self.level_set,
                        spacing=(grid['xyz'][0][2] - grid['xyz'][0][1],
                                 grid['xyz'][1][2] - grid['xyz'][1][1],
                                 grid['xyz'][2][2] - grid['xyz'][2][1]))
            except:
                print('marching_cubes error. Possibly no surface extracted from the level set.')
                return

            # convert back to world coordinates
            vertices = verts + np.array([grid['xyz'][0][0], grid['xyz'][1][0], grid['xyz'][2][0]])

            vertex_colors = None
            vertex_semantic = None

            if color:
                # color is extracted by passing the coordinates of mesh vertices through the network
                points = torch.from_numpy(vertices)
                z = []
                for i, pnts in enumerate(torch.split(points, self.points_batch_size, dim=0)):
                    z_color = self.eval_points(pnts.to(device).float(), all_planes, decoders, color=True).cpu()[..., :3]
                    z.append(z_color)
                z = torch.cat(z, dim=0)
                vertex_colors = z.numpy()

            if semantic:
                points = torch.from_numpy(vertices)
                z = []
                for i, pnts in enumerate(torch.split(points, self.points_batch_size, dim=0)):
                    z_semantic = self.eval_points(pnts.to(device).float(), all_planes, decoders, semantic=True).cpu()[..., :3]
                    z.append(z_semantic)
                z = torch.cat(z, dim=0)
                vertex_semantic = z.numpy()

            vertices /= self.scale
            if color:
                mesh_color = trimesh.Trimesh(vertices, faces, vertex_colors=vertex_colors)
                mesh_color.export(mesh_out_color)
            if semantic:
                mesh_semantic = trimesh.Trimesh(vertices, faces, vertex_colors=vertex_semantic)
                mesh_semantic.export(mesh_out_semantic)
                print('Saved mesh at', mesh_out_semantic)

            if self.verbose:
                print('Saved mesh at', mesh_out_semantic)

    def decode_segmap(self, image, nc):
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

        rgb = np.stack([r, g, b], axis=1)
        rgb = torch.from_numpy(rgb.astype(np.float32) / 255.0)  # convert to PyTorch tensor here

        return rgb