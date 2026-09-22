# [ETCC2026] DCAI-CISNet: Directional Chroma–Adaptive Intensity with Chromatic–Illumination Separation Network for Low-Light Image Enhancement

**[Mousumi Behera] (https://scholar.google.com/citations?user=XJnM_dEAAAAJ&hl=en&oi=ao)<sup>∗ </sup>, [Prasenjit Dey] (https://scholar.google.com/citations?user=Z46lTvcAAAAJ&hl=en&oi=ao)

## Pretrained Weights

All the weights that we trained on different datasets is available at [[Google Drive] (https://drive.google.com/drive/folders/1FOxEBY2fm83QLyAHhywGTQULJWh6y32D?usp=sharing)]

## 1. Get Started

### Dependencies and Installation

- Python 3.10
- PyTorch 2.5.1
- TorchVision 0.20.1
- CUDA 12.x
- NumPy 1.26.4

(1) Create Conda Environment

```bash
conda create --name CISNet python=3.10
conda activate CISNet
```
(2) Clone Repo

```bash
git clone https://github.com/mousumibehera821/DCAI-CISNet.git
cd DCAI-CISNet
```
(3) Install Dependencies
```bash
pip install -r requirements.txt
```
### Download the dataset

- [LOLv1] (https://drive.google.com/drive/folders/1vptcAgiR3uCbNNFCLBLlKTS3tX4n57NF?usp=sharing)
- [LOLv2] (https://drive.google.com/drive/folders/1eFQHkII2TfT9Vunt5OgbPDC-wI7i-2KK?usp=sharing)
- [DICM] (https://drive.google.com/drive/folders/1vPCIe043jb6xfgreuxL92iYFX-702aZ8?usp=sharing)
- [MEF] (https://drive.google.com/drive/folders/1rrjQHfKu78LgRl_iWuW35g37zyk5u5kl?usp=sharing)
- [LIME] (https://drive.google.com/drive/folders/1iUks0KLXxlYloYVCli324MvAOxJNUMnu?usp=sharing)
- [NPE] (https://drive.google.com/drive/folders/1B4uz2_Cmfmn_d5xA790J7Qg5Xd6I_Tel?usp=sharing)
- [VV] (https://drive.google.com/drive/folders/1cJFbCbvcN3n6SY2wmsFwkS4HnjeNXDoB?usp=sharing)

## 2. Testing
Download our weights from [[Google Drive] (https://drive.google.com/drive/folders/1FOxEBY2fm83QLyAHhywGTQULJWh6y32D?usp=sharing)]
```
├── weights
    ├── LOLv1.pth
    ├── LOLv2R.pth
    └── LOLv2S.pth
```
# LOLv1
python eval.py --lol

# LOLv2R
python eval.py --lol_v2_real --best_PSNR

# LOLv2S
python eval.py --lol_v2_syn

# DICM
python eval.py --unpaired --DICM

# MEF
python eval.py --unpaired --MEF

# LIME
python eval.py --unpaired --LIME

# NPE
python eval.py --unpaired --NPE

# VV
python eval.py --unpaired --VV

## 3. Training
- We put all the configurations that need to be adjusted in the `./data/options.py` folder.

```bash
# You can choose these dataset for training: lolv1, lolv2real, lolv2syn.
#Below is the example.
python train.py --dataset lol_v1
```
- All weights are saved to the `./weights/train` folder and are saved in steps of the checkpoint set in the `options.py`  as `epoch_*.pth` where `*` represent the epoch number.

## 4. Contact

- If you have any questions, please feel free to contact us or submit an issue to the repository!
    Mousumi Behera (mousumibehera821@gmail.com or 524cs3009@nitrkl.ac.in)

## 5. Citation
If you find our proposed work useful for your research, please cite our paper:

```bibtex
@inproceedings{Behera2026DCAICISNet,
  author    = {Mousumi Behera and Prasenjit Dey},
  title     = {DCAI-CISNet: Directional Chroma-Adaptive Intensity with Chromatic-Illumination Separation Network for Low-Light Image Enhancement},
  booktitle = {2026 International Conference on Emerging Technologies in Computing and Communication (ETCC)},
  year      = {2026},
  pages     = {1--6},
  doi       = {10.1109/ETCC69750.2026.11681954}
}
