# Comprehensive Knowledge Distillation with Causal Intervention

This repository is a PyTorch implementation of "Comprehensive Knowledge Distillation with Causal Intervention". The code is modified from [CRD], and the pretrained teachers (except WRN-40-4) are also downloaded from [CRD].

## Requirements

The code was tested on
```
Python 3.6
torch 1.2.0
torchvision 0.4.0
```

## Evaluation
To evaluate our pre-trained light-weight student networks, first download the folder "pretrained_student_model" from [CID models] into the "save" folder,  then simply run the command below to evaluate these light-weight students:

`sh evaluate_scripts.sh`

## Training
To train students from scratch by distilling knowledge from teacher networks with CID, first download the pretrained teacher folder "models" from [CID models] into the "save" folder, and then simply run the command below to compress large models to smaller ones:

`sh train_scripts.sh`

[CID models]: https://drive.google.com/drive/folders/1s-NwnDw3VXc_r87-XHEg1iM0KhxpXlbj?usp=sharing

[CRD]: https://github.com/HobbitLong/RepDistiller

## Citation
If you find this code helpful, you may consider citing this paper:
```bibtex
@inproceedings{deng2021comprehensive,
  title={Comprehensive Knowledge Distillation with Causal Intervention},
  author={Deng, Xiang and Zhang, Zhongfei},
  booktitle = {Proceedings of the 30th Annual Conference on Neural Information Processing Systems},
  year={2021}
}
```
