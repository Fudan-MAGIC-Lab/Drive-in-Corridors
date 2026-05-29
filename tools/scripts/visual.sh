CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0 python3 -m torch.distributed.launch --master_port=29504 \
./tools/visual_pred_nusc.py projects/configs/VAD/Corridor_stage_2.py \
corridor_epoch_48_learned_weight_12.pth --launcher pytorch --samples 10
# epoch_normlized_12.pth --launcher pytorch