# This file is a part of SNI-SLAM.

from src.networks.decoders import Decoders

def get_model(cfg):
    c_dim = cfg['model']['c_dim']  # feature dimensions
    truncation = cfg['model']['truncation']
    learnable_beta = cfg['rendering']['learnable_beta']
    hidden_dim = cfg['model']['hidden_dim']
    sem_hidden_dim = cfg['model']['sem_hidden_dim']
    fused_hidden_dim = cfg['model']['fused_hidden_dim']

    num_classes = cfg['model']['cnn']['n_classes']


    decoder = Decoders(c_dim=c_dim, hidden_dim=hidden_dim, truncation=truncation, learnable_beta=learnable_beta,
                       num_classes=num_classes, sem_hidden_dim=sem_hidden_dim, fused_hidden_dim=fused_hidden_dim)

    return decoder
