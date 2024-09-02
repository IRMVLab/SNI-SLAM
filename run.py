# This file is a part of SNI-SLAM
#
import argparse

from src import config
from src.SNI_SLAM import SNI_SLAM

def main():
    parser = argparse.ArgumentParser(
        description='Arguments for running SNI_SLAM.'
    )
    parser.add_argument('config', type=str, help='Path to config file.')
    parser.add_argument('--input_folder', type=str,
                        help='input folder, this have higher priority, can overwrite the one in config file')
    parser.add_argument('--output', type=str,
                        help='output folder, this have higher priority, can overwrite the one in config file')
    args = parser.parse_args()

    cfg = config.load_config(args.config, 'configs/SNI-SLAM.yaml')
    sni_slam = SNI_SLAM(cfg, args)

    sni_slam.run()

if __name__ == '__main__':
    main()
