CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run --nproc_per_node=4 --master_port=2345\
 tools/train.py projects/configs/VAD/Corridor_stage_2.py --launcher pytorch --deterministic --work-dir work_dir\
 --load-from corridor_epoch_48.pth