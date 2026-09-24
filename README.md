# E³SAM2

[E³SAM2: Entropy-Aware and Edge-Guided Adaptation of SAM2 for Echocardiography Video Segmentation](https://doi.org/10.1609/aaai.v40i16.38346), AAAI 2026.

[Paper](https://ojs.aaai.org/index.php/AAAI/article/view/38346)

Long Zheng, Zhi Li, Weidong Wang, Zhenyu Dai, and Shuyun Li

E³SAM2 is a SAM 2-based framework for echocardiography video segmentation. This repository supports semi-supervised and fully supervised CAMUS experiments, as well as EchoNet-Dynamic video segmentation.

## Installation

Python 3.10 or later and a CUDA-capable GPU are recommended.

```bash
conda create -n e3sam2 python=3.10 -y
conda activate e3sam2
pip install -r requirements.txt
```

## Usage

### Prepare datasets

Download the datasets from their official websites:

- [CAMUS](https://www.creatis.insa-lyon.fr/Challenge/camus/)
- [EchoNet-Dynamic](https://echonet.github.io/dynamic/)

Generate the resized videos, segmentation masks, and edge targets with the preprocessing scripts:

```bash
# CAMUS
python utils/preprocess_camus_edge.py \
  --input-dir /path/to/CAMUS/database_nifti \
  --output-dir /path/to/CAMUS_Processed

# EchoNet-Dynamic
python utils/preprocess_echonet_edge.py \
  --input-dir /path/to/EchoNet-Dynamic \
  --output-dir /path/to/EchoNet_Processed
```

CAMUS experiments use the `7-1-2` split reported in the paper. EchoNet-Dynamic uses its official split.

### Download the pretrained checkpoint

Download the SAM 2 Hiera-S checkpoint from the [official SAM 2 repository](https://github.com/facebookresearch/sam2) and place it at:

```text
pretrained_checkpoints/sam2_hiera_small.pt
```

Another checkpoint location can be specified with `--pretrained`.

### Download trained checkpoints

The trained E³SAM2 checkpoints are available in the
[v1.0.0 release](https://github.com/ZhengLong777/E3SAM2/releases/tag/v1.0.0):

- [CAMUS semi-supervised checkpoint](https://github.com/ZhengLong777/E3SAM2/releases/download/v1.0.0/best_camus_semi.pth)
- [EchoNet-Dynamic checkpoint](https://github.com/ZhengLong777/E3SAM2/releases/download/v1.0.0/best_echonet.pth)

### Train

```bash
python train.py \
  --task CAMUS_Video_Semi \
  --epochs 200 \
  --gpu 0 \
  --data /path/to/CAMUS_Processed
```

Available tasks are `CAMUS_Video_Full`, `CAMUS_Video_Semi`, and `EchoNet_Video`. Use `--mask-loss` to select the segmentation loss.

### Test

```bash
python test.py \
  --task CAMUS_Video_Semi \
  --checkpoint /path/to/best.pth \
  --data /path/to/CAMUS_Processed
```

Run `python train.py --help` or `python test.py --help` for all options.

## Acknowledgement

This codebase is developed substantially based on [MemSAM](https://github.com/dengxl0520/MemSAM). We sincerely thank the MemSAM authors for releasing their code for echocardiography video segmentation.

The model implementation also builds on [SAM 2](https://github.com/facebookresearch/sam2) and benefits from the medical image and video segmentation work in [MedSAM2](https://github.com/bowang-lab/MedSAM2). We thank the authors of MemSAM, MedSAM2, SAM 2, and their upstream projects for their open-source contributions.

## License

This project is released under the [Apache License 2.0](LICENSE). Third-party
components remain subject to their original copyright and license terms. See
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for details.

## Citation

If you find this repository useful, please cite our E³SAM2 paper, thank you!

```bibtex
@article{zheng2026e3sam2,
    author  = {Zheng, Long and Li, Zhi and Wang, Weidong and Dai, Zhenyu and Li, Shuyun},
    title   = {{E³SAM2}: Entropy-Aware and Edge-Guided Adaptation of SAM2 for Echocardiography Video Segmentation},
    journal = {Proceedings of the AAAI Conference on Artificial Intelligence},
    volume  = {40},
    number  = {16},
    pages   = {13423--13431},
    year    = {2026},
    doi     = {10.1609/aaai.v40i16.38346}
}
```

Please also cite MemSAM:

```bibtex
@InProceedings{Deng_2024_CVPR,
    author    = {Deng, Xiaolong and Wu, Huisi and Zeng, Runhao and Qin, Jing},
    title     = {MemSAM: Taming Segment Anything Model for Echocardiography Video Segmentation},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2024},
    pages     = {9622--9631}
}
```

Please also cite MedSAM2 and SAM 2:

```bibtex
@article{MedSAM2,
    title   = {MedSAM2: Segment Anything in 3D Medical Images and Videos},
    author  = {Ma, Jun and Yang, Zongxin and Kim, Sumin and Chen, Bihui and Baharoon, Mohammed and Fallahpour, Adibvafa and Asakereh, Reza and Lyu, Hongwei and Wang, Bo},
    journal = {arXiv preprint arXiv:2504.03600},
    year    = {2025}
}

@inproceedings{SAM2,
    title     = {{SAM} 2: Segment Anything in Images and Videos},
    author    = {Ravi, Nikhila and Gabeur, Valentin and Hu, Yuan-Ting and Hu, Ronghang and Ryali, Chaitanya and Ma, Tengyu and Khedr, Haitham and R{\"a}dle, Roman and Rolland, Chloe and Gustafson, Laura and Mintun, Eric and Pan, Junting and Alwala, Kalyan Vasudev and Carion, Nicolas and Wu, Chao-Yuan and Girshick, Ross and Dollar, Piotr and Feichtenhofer, Christoph},
    booktitle = {International Conference on Learning Representations},
    year      = {2025}
}
```
