<!-- PROJECT LOGO -->

<p align="center">

  <h1 align="center">VPGS-SLAM: Voxel-based Progressive 3D Gaussian
SLAM in Large-Scale Scenes</h1>
  <div align="center"></div>
</p>

<p align="center">
  <a href="">
    <img src="Fig/framework.png" alt="Logo" width="100%">
  </a>
</p>

<br>
<!-- TABLE OF CONTENTS -->
<details open="open" style='padding: 10px; border-radius:5px 30px 30px 5px; border-style: solid; border-width: 1px;'>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#installation">Installation</a>
    </li>
    <li>
      <a href="#demo">Online Demo</a>
    </li>
    <li>
      <a href="#usage">Usage</a>
    </li>
    <li>
      <a href="#downloads">Downloads</a>
    </li>
    <li>
      <a href="#benchmarking">Benchmarking</a>
    </li>
    <li>
      <a href="#acknowledgement">Acknowledgement</a>
    </li>
    <li>
      <a href="#citation">Citation</a>
    </li>
  </ol>
</details>

# Notes

Our method is a plug-and-play approach that can be integrated with different GS-SLAM frameworks. We will maintain separate branches for different versions of the method. The current version supports MonoGS and Gaussian-SLAM.

- [x] We have updated the `README.md` and are preparing to open-source our code！  
- [x] Code for main parts, including `optimizer`, `renderer`, `mapping modules`, `Voxelized Scene Representation`
- [x] Installation setup
- [ ] 2D-3D Tracking Module
# Setup
The code has been tested on:

- Ubuntu 22.04 LTS, Python 3.10.14, CUDA 12.2, GeForce RTX 4090/RTX 3090
- CentOS Linux 7, Python 3.12.1, CUDA 12.4, A100/A6000

## Repository

Clone the repo with `--recursive` because we have submodules:

```
git clone --recursive git@github.com/dtc111111/vpgs-slam.git
cd VPGS-SLAM
```

## Installation
Make sure that gcc and g++ paths on your system are exported:

```
export CC=<gcc path>
export CXX=<g++ path>
```

To find the <i>gcc path</i> and <i>g++ path</i> on your machine, you can use <i>which gcc</i>.


Then setup environment from the provided conda environment file,

```
conda create -n vpgs-slam -c nvidia/label/cuda-12.1.0 cuda=12.1 cuda-toolkit=12.1 cuda-nvcc=12.1
conda env update --file environment.yml --prune
conda activate vpgs-slam
pip install -r requirements.txt
```

You will also need to install <i>hloc</i> for loop detection and 3DGS registration.
```
cd thirdparty/Hierarchical-Localization
python -m pip install -e .
cd ../..
```

We tested our code on RTX4090 and RTX A6000 GPUs respectively and Ubuntu22 and CentOS7.5.

## 🚀 Usage

Here we elaborate on how to load the necessary data, configure Gaussian-SLAM for your use-case, debug it, and how to reproduce the results mentioned in the paper.

  <!-- <details>
  <summary><b>Downloading the Data</b></summary> -->
  ### Downloading the Datasets
  We tested our code on Replica, TUM_RGBD, ScanNet, and ScanNet++ datasets. We also provide scripts for downloading Replica and TUM_RGBD in `scripts` folder. Install git lfs before using the scripts by running ```git lfs install```.

  For reconstruction evaluation on <b>Replica</b>, we follow [Co-SLAM](https://github.com/JingwenWang95/neural_slam_eval?tab=readme-ov-file#datasets) mesh culling protocal, please use their code to process the mesh first.

  For downloading ScanNet, follow the procedure described on <a href="http://www.scan-net.org/">here</a>.<br>
  <b>Pay attention! </b> There are some frames in ScanNet with `inf` poses, we filter them out using the jupyter notebook `scripts/scannet_preprocess.ipynb`. Please change the path to your ScanNet data and run the cells.

  For downloading ScanNet++, follow the procedure described on <a href="https://kaldir.vc.in.tum.de/scannetpp/">here</a>.<br>
  
  The config files are named after the sequences that we used for our method.
  <!-- </details> -->

  ### Running the code
  Start the system with the command:

  ```
  python run_slam.py configs/<dataset_name>/<config_name> --input_path <path_to_the_scene> --output_path <output_path>
  ```
  
  You can also configure input and output paths in the config yaml file.
  
  


# ✏️ Acknowledgement
Our implementation is heavily based on <a href="https://vladimiryugay.github.io/gaussian_slam/index.html">Gaussian-SLAM</a> and <a href="https://github.com/muskie82/MonoGS">MonoGS</a> and <a href="https://github.com/GradientSpaces/LoopSplat">Loop-Splat</a>. We thank the authors for their open-source contributions. If you use the code that is based on their contribution, please cite them as well.

