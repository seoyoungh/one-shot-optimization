# Prediction-based One-shot Dynamic Parking Pricing
## Introduction
This is the repository of our accepted CIKM 2022 paper "Prediction-based One-shot Dynamic Parking Pricing". Paper is available on [arxiv](). You can also download data [here](https://drive.google.com/drive/folders/1M0rzyiOHNzTMBdXMsDuQlSQRLqLerXX9?usp=sharing).

## Citation
If you find this code useful, you may cite us as:
```
@article{hong2022prediction,
  title={Prediction-based One-shot Dynamic Parking Pricing},
  author={Hong, Seoyoung and Shin, Heejoo and Choi, Jeongwhan and Park, Noseong},
  journal={arXiv preprint arXiv:2208.14231},
  year={2022}
}
```

## Setup an environment
```
$ conda env create -f requirements.yaml 
```

## Usage
### i) train parking occupancy rate prediction model
    - Run run_*.sh to train the prediction model or just pass as we uploaded the pre-trained model.
### ii) optimize parking price with pre-trained prediction model
    - Run optimize_*.sh to optimize the price with the pre-trained prediction model.