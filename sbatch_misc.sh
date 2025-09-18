#!/bin/bash

#SBATCH --job-name append-calvin-data
#SBATCH --requeue
##SBATCH --exclude=dlc2gpu10,dlc2gpu07
#SBATCH --signal=SIGUSR1@600
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=320G
#SBATCH --time=1-00:00
##SBATCH --partition rldlc2_gpu-l40s
##SBATCH --partition alldlc2_gpu-l40s
##SBATCH --partition alldlc2_gpu-h200
#SBATCH --partition alldlc_gpu-rtx2080
##SBATCH --partition aisdlc_gpu-rtx2080



#SBATCH --output=/work/dlclarge1/lagandua-vlawm/lumos/logs/outputs/%x_%j.out
#SBATCH --error=/work/dlclarge1/lagandua-vlawm/lumos/logs/outputs/%x_%j.err

# log basic job/cluster info
echo "== SLURM INFO =========================================================================================================="
echo "time:                     $(date)"
echo "directory:                ${PWD}"
echo "name:                     ${SLURM_JOB_NAME}"
echo "partition:                ${SLURM_JOB_PARTITION}"
echo "nodes:                    ${SLURM_NODELIST}"
echo "tasks per node:           ${SLURM_TASKS_PER_NODE}"
echo "cpus per task:            ${SLURM_CPUS_PER_TASK}"
echo ""
echo "== ENVIRONMENT ========================================================================================================="
echo "CUDA_VISIBLE_DEVICES:     ${CUDA_VISIBLE_DEVICES}"
echo ""
echo "== SETUP ==============================================================================================================="

# set file limit
file_limit=$(ulimit -Hn)
echo "Hard file limit is set at ${file_limit}, setting soft limit to match this"
ulimit -n ${file_limit}
echo "Set soft file limit to ${file_limit}"

# prepare and run
export NCCL_P2P_DISABLE=1
export PYTHONFAULTHANDLER=1
export TORCH_USE_CUDA_DSA=1

echo ""
echo "== LOG ================================================================================================================="

# Activate the virtualenv / conda environment
source /home/lagandua/.bashrc
conda activate lumos


srun python /work/dlclarge1/lagandua-vlawm/lumos/scripts/append_calvin_patch_dataset.py
# batch_size=2 model=smolflow_agent # batch_size=4 model.use_lora=False # model=vlm_berg_agent # batch_size=2 model.vla_mode='reduced_head' #model.use_perceiver=False model.use_incontext=True #seed=242 #model=mode_agent