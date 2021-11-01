python train_student.py --path_t './save/models/wrn_40_4_vanilla/ckpt_epoch_240.pth' --model_s wrn_16_2 -NT 4 -a 1 -b 55 -c 0.1
python train_student.py --path_t './save/models/wrn_40_2_vanilla/ckpt_epoch_240.pth' --model_s wrn_16_2 -NT 4 -a 1.0 -b 65 -c 1.0
python train_student.py --path_t './save/models/resnet56_vanilla/ckpt_epoch_240.pth' --model_s resnet20 -NT 4 -a 1 -b 10 -c 1
python train_student.py --path_t './save/models/ResNet50_vanilla/ckpt_epoch_240.pth' --model_s MobileNetV2 -NT 4 -a 1.0 -b 33.0 -c 0.1
python train_student.py --path_t './save/models/ResNet50_vanilla/ckpt_epoch_240.pth' --model_s vgg8 -NT 4 -a 1.0 -b 8 -c 2.0
