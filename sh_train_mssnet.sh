#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1,2,3 python run_dp.py \
                --train_datalist './datalist/RealBlur_R_train_list.txt'\
                --data_root_dir './dataset'\
                --checkdir './checkpoint/SDOR_Realblur_r600'\
                --max_epoch 600\
                --wf 54\
                --scale 42\
                --vscale 42\
                --mgpu
