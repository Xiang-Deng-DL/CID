#evalaute pretrained student models
python evaluate_student.py --model_path './save/pretrained_student_model/T_wrn40_4_S_wrn_16_2_cifar100_CID/ckpt_epoch_240.pth' --model_s wrn_16_2
python evaluate_student.py --model_path './save/pretrained_student_model/T_wrn40_2_S_wrn_16_2_cifar100_CID/ckpt_epoch_240.pth' --model_s wrn_16_2
python evaluate_student.py --model_path './save/pretrained_student_model/T_resnet56_S_resnet20_cifar100_CID/ckpt_epoch_240.pth' --model_s resnet20
python evaluate_student.py --model_path './save/pretrained_student_model/T_ResNet50_S_MobileNetV2_cifar100_CID/ckpt_epoch_240.pth' --model_s MobileNetV2
python evaluate_student.py --model_path './save/pretrained_student_model/T_ResNet50_S_vgg8_cifar100_CID/ckpt_epoch_240.pth' --model_s vgg8