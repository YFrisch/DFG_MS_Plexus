CHCKPT='/home/yfrisch_locale/nnUNet/nnUNet_results/Dataset001_DFGPlexus/nnUNetTrainer_500epochs__nnUNetResEncUNetMPlans__2d/fold_all/checkpoint_best.pth'
CUDA_LAUNCH_BLOCKING=1 nnUNet_compile=False nnUNetv2_train 002 2d 0 -p nnUNetResEncUNetMPlans -pretrained_weights $CHCKPT -tr nnUNetTrainer_500epochs --npz &&
CUDA_LAUNCH_BLOCKING=1 nnUNet_compile=False nnUNetv2_train 002 2d 1 -p nnUNetResEncUNetMPlans -pretrained_weights $CHCKPT -tr nnUNetTrainer_500epochs --npz &&
CUDA_LAUNCH_BLOCKING=1 nnUNet_compile=False nnUNetv2_train 002 2d 2 -p nnUNetResEncUNetMPlans -pretrained_weights $CHCKPT -tr nnUNetTrainer_500epochs --npz &&
CUDA_LAUNCH_BLOCKING=1 nnUNet_compile=False nnUNetv2_train 002 2d 3 -p nnUNetResEncUNetMPlans -pretrained_weights $CHCKPT -tr nnUNetTrainer_500epochs --npz &&
CUDA_LAUNCH_BLOCKING=1 nnUNet_compile=False nnUNetv2_train 002 2d 4 -p nnUNetResEncUNetMPlans -pretrained_weights $CHCKPT -tr nnUNetTrainer_500epochs --npz