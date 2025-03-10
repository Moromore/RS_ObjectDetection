#!/bin/bash -l

#SBATCH --output=slurm-test/test-1iter-test.out
#SBATCH --error=slurm-test/test-1iter-test.err
#SBATCH --partition=gpu    # partition name
#SBATCH --qos=standard
#SBATCH --nodes=1               # Total number of nodes
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-node=4
#SBATCH --time=1:00:00         # Run time (d-hh:mm:ss)

module load cuda/11.4.1
export CXX=/usr/bin/g++
source ../miniconda3/bin/activate obd-2

start_time=$(date)
echo "Job started at: $start_time"


#CUDA_VISIBLE_DEVICES=0 python -u tools/train.py configs/obb/oriented_rcnn/vit_base_win/rs5m_base.py \
#> slurm-test/rs5m_base_lr1e-4.out 2> slurm-test/rs5m_base_lr1e-4.err &

#CUDA_VISIBLE_DEVICES=1 python -u tools/train.py configs/obb/oriented_rcnn/vit_base_win/rs5m_huge.py \
#> slurm-test/rs5m_huge_lr1e-4.out 2> slurm-test/rs5m_huge_lr1e-4.err &

#CUDA_VISIBLE_DEVICES=2 python -u tools/train.py configs/obb/oriented_rcnn/vit_base_win/remoteclip_base.py \
#> slurm-test/remoteclip_base_lr1e-4.out 2> slurm-test/remoteclip_base_lr1e-4.err &

#CUDA_VISIBLE_DEVICES=3 python -u tools/train.py configs/obb/oriented_rcnn/vit_base_win/remoteclip_large.py \
#> slurm-test/remoteclip_large_lr1e-4.out 2> slurm-test/remoteclip_large_lr1e-4.err &

#CUDA_VISIBLE_DEVICES=1 python -u tools/train.py configs/obb/oriented_rcnn/vit_base_win/SatMAEpp_large.py \
#> slurm-test/SatMAEpp_large-1iter.out 2> slurm-test/SatMAEpp_large-1iter.err &
export CUDA_VISIBLE_DEVICES=0,1,2,3
srun python -u tools/test.py configs/obb/oriented_rcnn/vit_base_win/SatMAEpp_large.py \
work_dirs/SatMAEpp/SatMAEpp_large/latest.pth \
--out work_dirs/save/SatMAEpp_large/det_result.pkl \
--eval 'mAP' \
--show-dir 'work_dirs/save/SatMAEpp_large/display/' \
--launcher="slurm"
#
#CUDA_VISIBLE_DEVICES=1 python -u tools/train.py configs/obb/oriented_rcnn/vit_base_win/rsva_base.py \
#> slurm-test/rsva_base.out 2> slurm-test/rsva_base.err &
#
#CUDA_VISIBLE_DEVICES=2 python -u tools/train.py configs/obb/oriented_rcnn/vit_base_win/scale-MAE_large.py \
#> slurm-test/scale-MAE_large.out 2> slurm-test/scale-MAE_large.err &
#
#CUDA_VISIBLE_DEVICES=3 python -u tools/train.py configs/obb/oriented_rcnn/vit_base_win/SatMAE_DIOR.py \
#> slurm-test/SatMAE_DIOR_lr_5e-5.out 2> slurm-test/SatMAE_DIOR_lr_5e-5.err &

wait

end_time=$(date)
echo "Job ended at: $end_time"
total_seconds=$(($(date +%s -d "$end_time") - $(date +%s -d "$start_time")))
hours=$((total_seconds / 3600))
minutes=$(( (total_seconds % 3600) / 60 ))
seconds=$((total_seconds % 60))
echo "Total runtime: ${hours}h ${minutes}m ${seconds}s"

#export CUDA_VISIBLE_DEVICES=0,1,2,3
#srun python -u tools/test.py configs/obb/oriented_rcnn/vit_base_win/SatMAE_DIOR.py \
#work_dirs/SatMAE/SatMAE_DIOR/latest.pth \
#--out work_dirs/save/faster/full_det/SatMAE_DIOR.py/det_result.pkl \
#--eval 'mAP' \
#--show-dir work_dirs/save/faster/display/SatMAE_DIOR.py \
#--launcher="slurm"