# This file is a part of SNI-SLAM.

from src.networks.dinov2_seg import DINO2SEG
from src.networks.mlp import head, encoder, feature_fusion
import torch

class ModelManager:
    def __init__(self, cfg):
        self.dim = cfg['model']['c_dim']
        self.hidden_dim = cfg['model']['hidden_dim']

        self.encoder_multires = cfg['model']['encoder']['multires']

        self.pretrained_model_path = cfg['model']['cnn']['pretrained_model_path']
        self.n_classes = cfg['model']['cnn']['n_classes']

        self.img_h = cfg['cam']['H']
        self.img_w = cfg['cam']['W']

        self.crop_edge = cfg['cam']['crop_edge']

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.encoder = self.get_encoder().cuda()
        self.head = self.get_head().cuda()
        self.cnn = self.get_dinov2().cuda()

        self.feat_fusion = self.get_fusion().cuda()

    def train(self):
        self.feat_fusion.train()
        self.encoder.train()
        self.head.train()

    def get_encoder(self):
        return encoder(hidden_dim=self.hidden_dim, out_dim=self.dim, multires=self.encoder_multires)

    def get_head(self):
        return head(in_dim=self.dim, hidden_dim=self.hidden_dim)

    def get_dinov2(self):
        model = DINO2SEG(img_h=self.img_h, img_w=self.img_w, num_cls=self.n_classes, edge=self.crop_edge, dim=self.dim)
        model.load_state_dict(torch.load(self.pretrained_model_path, map_location=self.device))
        return model

    def set_mode_feature(self):
        self.cnn.mode = 'mapping'

    def set_mode_result(self):
        self.cnn.mode = 'train'

    def get_fusion(self):
        return feature_fusion(dim=self.dim, hidden_dim=self.hidden_dim)

    def get_share_memory(self):
        self.feat_fusion.share_memory()
        self.encoder.share_memory()
        self.head.share_memory()
        self.cnn.share_memory()

        return self