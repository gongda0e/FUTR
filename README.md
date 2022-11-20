# Future Transformer for Long-term Action Anticipation (CVPR 2022)
### [Project Page](http://cvlab.postech.ac.kr/research/FUTR/) | [Paper](https://arxiv.org/abs/2205.14022)
This repository contains the official source code and data for our paper:

> [Future Transformer for Long-term Action Anticipation](https://arxiv.org/abs/2205.14022)  
> [Dayoung Gong](https://gongda0e.github.io/),
> [Joonseok Lee](https://scholar.google.com/citations?user=ZXcSl7cAAAAJ&hl=ko),
> [Manjin Kim](https://kimmanjin.github.io/),
> [Seong Jong Ha](https://scholar.google.co.kr/citations?user=hhQc51AAAAAJ&hl=ko), and
> [Minsu Cho](http://cvlab.postech.ac.kr/~mcho/)
> POSTECH & NCSOFT
> CVPR, New Orleans, 2022.

<div style="text-align:center">
<img src="pipeline.png" alt="An Overview of the proposed pipeline"/>
</div>



## Environmental setup
* Conda environment settings:
```bash
conda env export > futr.yaml
conda activate futr
```

## Dataset
Download the data from https://mega.nz/file/O6wXlSTS#wcEoDT4Ctq5HRq_hV-aWeVF1_JB3cacQBQqOLjCIbc8 .   
Create a directory './datasets' for the two datasets and place each dataset to have following directory structure:
```bash
    ../                         # parent directory
    ├── ./                      # current (project) directory
    │   ├── data/               # (dir.) dataloaders for action anticipation dataset
    │   ├── model/              # (dir.) implementation of Hypercorrelation Squeeze Network model 
    │   ├── README.md           # intstruction for reproduction
    │   ├── train.py            # code for training FUTR
    │   ├── predict.py          # code for testing FUTR
    │   ├── otps.py             # code for arguments
    │   └── utils.py            # code for helper functions
    └── datasets/
        ├── breakfast/          # Breakfast dataset
        │   ├── groundTruth/
        │   ├── features/
        │   └── ...
        ├── 50salads/          # 50salads dataset
        │   ├── groundTruth/
        │   ├── features/
        │   └── ...
```

## Training
> ### 1. Breakfast
> ```bash
>./scripts/train.sh $split_num
>```

> ### 2. 50salads
> ```bash
>./scripts/50s_train.sh $split_num
>```

## Testing
> ### 1. Breakfast
> ```bash
>./scripts/predict.sh $split_num
>```

> ### 2. 50salads
> ```bash
>./scripts/50s_predict.sh $split_num
>```

## Citation
If you find our code or paper useful, please consider citing our paper:
```BibTeX
@inproceedings{gong2022future,
  title={Future Transformer for Long-term Action Anticipation},
  author={Gong, Dayoung and Lee, Joonseok and Kim, Manjin and Ha, Seong Jong and Cho, Minsu},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={3052--3061},
  year={2022}
}
```
