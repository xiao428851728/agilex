# AgilexNav

Zero-shot semantic object navigation. The agent builds an occupancy map from depth observations to find exploration frontiers, uses RGB observations with a pre-trained vision-language model to build a language-grounded value map, and picks the most promising frontier to explore toward the target object category. The code supports simulation experiments in Habitat (HM3D / MP3D / Gibson) and deployment on a real mobile robot.

## :hammer_and_wrench: Installation

### Getting Started
Create the conda environment:
```bash
conda_env_name=agilex_nav
conda create -n $conda_env_name python=3.9 -y
conda activate $conda_env_name
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 -f https://download.pytorch.org/whl/torch_stable.html
pip install git+https://github.com/IDEA-Research/GroundingDINO.git@eeba084341aaa454ce13cb32fa7fd9282fc73a67 salesforce-lavis==1.0.2
```
If you are using habitat and are doing simulation experiments, install this repo into your env with the following:
```bash
pip install -e .[habitat]
```
If you are using the Spot robot, install this repo into your env with the following:
```bash
pip install -e .[reality]
```
#### [Whether you're using conda or not]
Clone the following repo within this one (simply cloning will suffice):
```bash
git clone git@github.com:WongKinYiu/yolov7.git
```

## :dart: Downloading the HM3D dataset

### Matterport
First, set the following variables during installation (don't need to put in .bashrc):
```bash
MATTERPORT_TOKEN_ID=<FILL IN FROM YOUR ACCOUNT INFO IN MATTERPORT>
MATTERPORT_TOKEN_SECRET=<FILL IN FROM YOUR ACCOUNT INFO IN MATTERPORT>
DATA_DIR=</path/to/agilex_nav/data>

# Link to the HM3D ObjectNav episodes dataset, listed here:
# https://github.com/facebookresearch/habitat-lab/blob/main/DATASETS.md#task-datasets
# From the above page, locate the link to the HM3D ObjectNav dataset.
# Verify that it is the same as the next two lines.
HM3D_OBJECTNAV=https://dl.fbaipublicfiles.com/habitat/data/datasets/objectnav/hm3d/v1/objectnav_hm3d_v1.zip
```

### Clone and install habitat-lab, then download datasets
*Ensure that the correct conda environment is activated!!*
```bash
# Download HM3D 3D scans (scenes_dataset)
python -m habitat_sim.utils.datasets_download \
  --username $MATTERPORT_TOKEN_ID --password $MATTERPORT_TOKEN_SECRET \
  --uids hm3d_train_v0.2 \
  --data-path $DATA_DIR &&
python -m habitat_sim.utils.datasets_download \
  --username $MATTERPORT_TOKEN_ID --password $MATTERPORT_TOKEN_SECRET \
  --uids hm3d_val_v0.2 \
  --data-path $DATA_DIR &&

# Download HM3D ObjectNav dataset episodes
wget $HM3D_OBJECTNAV &&
unzip objectnav_hm3d_v1.zip &&
mkdir -p $DATA_DIR/datasets/objectnav/hm3d  &&
mv objectnav_hm3d_v1 $DATA_DIR/datasets/objectnav/hm3d/v1 &&
rm objectnav_hm3d_v1.zip
```

## :weight_lifting: Downloading weights for various models
The weights for MobileSAM, GroundingDINO, and PointNav must be saved to the `data/` directory. The weights can be downloaded from the following links:
- `mobile_sam.pt`:  https://github.com/ChaoningZhang/MobileSAM
- `groundingdino_swint_ogc.pth`: https://github.com/IDEA-Research/GroundingDINO
- `yolov7-e6e.pt`: https://github.com/WongKinYiu/yolov7
- `pointnav_weights.pth`: included inside the [data](data) subdirectory

## :arrow_forward: Evaluation within Habitat
To run evaluation, various models must be loaded in the background first. This only needs to be done once by running the following command:
```bash
./scripts/launch_vlm_servers.sh
```
(You may need to run `chmod +x` on this file first.)
This command will create a tmux session that will start loading the various models used for AgilexNav and serving them through `flask`. When you are done, be sure to kill the tmux session to free up your GPU.

Run the following to evaluate on the HM3D dataset:
```bash
python -m agilex_nav.run
```
To evaluate on MP3D, run the following:
```bash
python -m agilex_nav.run habitat.dataset.data_path=data/datasets/objectnav/mp3d/val/val.json.gz
```
## :newspaper: License

Released under the [MIT License](LICENSE).
