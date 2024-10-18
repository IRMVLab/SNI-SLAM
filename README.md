# SNI-SLAM: Semantic Neural Implicit SLAM
Siting Zhu*, Guangming Wang*, Hermann Blum, Jiuming Liu, Liang Song, Marc Pollefeys, Hesheng Wang
<div align="center">
  <h3>CVPR 2024 [<a href="https://arxiv.org/pdf/2311.11016.pdf">Paper</a>] [<a href="https://drive.google.com/file/d/1oRKoly8cxple0Z3CcgbBvC_8wYQhOtR3/view?usp=drive_link">Suppl</a>]</h3>
</div>

## Demo

<p align="center">
  <a href="">
    <img src="./demo/sem_mapping.gif" alt="Logo" width="80%">
  </a>
</p>

## Installation

First you have to make sure that you have all dependencies in place.
The simplest way to do so, is to use [anaconda](https://www.anaconda.com/). 

You can create an anaconda environment called `sni`. For linux, you need to install **libopenexr-dev** before creating the environment.
```bash
sudo apt-get install libopenexr-dev
conda env create -f environment.yaml
conda activate sni
```

## Run
### Replica
1. Download the data with semantic annotations in [google drive](https://drive.google.com/drive/u/0/folders/1BCu8bCGKG9HmnLFbyx7DIHI0slgkeo4h) and save the data into the `./data/replica` folder. We only provide a subset of Replica dataset. For all Replica data generation, please refer to directory `data_generation`. 
2. Download the pretrained segmentation network in [google drive](https://drive.google.com/drive/u/0/folders/1BCu8bCGKG9HmnLFbyx7DIHI0slgkeo4h) and save it into the `./seg` folder.
and you can run SNI-SLAM:
```bash
python -W ignore run.py configs/Replica/room1.yaml
```
The mesh for evaluation is saved as `$OUTPUT_FOLDER/mesh/final_mesh_eval_rec_culled.ply`


## Evaluation

### Average Trajectory Error
To evaluate the average trajectory error. Run the command below with the corresponding config file:
```bash
# An example for room1 of Replica
python src/tools/eval_ate.py configs/Replica/room1.yaml
```

## Visualizing SNI-SLAM Results
For visualizing the results, we recommend to set `mesh_freq: 40` in [configs/SNI-SLAM.yaml](configs/SNI-SLAM.yaml) and run SNI-SLAM from scratch.

After SNI-SLAM is trained, run the following command for visualization.

```bash
python visualizer.py configs/Replica/room1.yaml --top_view --save_rendering
```
The result of the visualization will be saved at `output/Replica/room1/vis.mp4`. The green trajectory indicates the ground truth trajectory, and the red one is the trajectory of SNI-SLAM.

### Visualizer Command line arguments
- `--output $OUTPUT_FOLDER` output folder (overwrite the output folder in the config file)
- `--top_view` set the camera to top view. Otherwise, the camera is set to the first frame of the sequence
- `--save_rendering` save rendering video to `vis.mp4` in the output folder
- `--no_gt_traj` do not show ground truth trajectory

## Citing
If you find our code or paper useful, please consider citing:
```BibTeX
@inproceedings{zhu2024sni,
  title={Sni-slam: Semantic neural implicit slam},
  author={Zhu, Siting and Wang, Guangming and Blum, Hermann and Liu, Jiuming and Song, Liang and Pollefeys, Marc and Wang, Hesheng},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={21167--21177},
  year={2024}
}
```