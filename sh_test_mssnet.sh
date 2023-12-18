##!/bin/bash
#
#CUDA_VISIBLE_DEVICES=0 python test.py \
#                --test_datalist './datalist/datalist_hide_testset.txt'\
#                --data_root_dir './dataset/RealBlur_J/'\
#                --load_dir './checkpoint/SRDO/model_03000E.pt'\
#                --outdir './result/SRDO_Realblur_J'\
#                --wf 54\
#                --scale 42\
#                --vscale 42\
#                --is_eval\
#                --is_save
#/bin/bash
CUDA_VISIBLE_DEVICES=0 python test.py \
                --test_datalist './datalist/datalist_gopro_testset.txt'\
                --data_root_dir './dataset'\
                --load_dir './checkpoint/MSSNet/model_03000E.pt'\
                --outdir './result/MSSNet_ys'\
                --wf 54\
                --scale 42\
                --vscale 42\
                --is_eval\
                --is_save