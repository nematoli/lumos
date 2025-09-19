#!/bin/bash

#SBATCH --job-name train-patch-wm
#SBATCH --requeue
##SBATCH --exclude=dlc2gpu10,dlc2gpu07
#SBATCH --signal=SIGUSR1@600
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=320G
#SBATCH --time=1-00:00
#SBATCH --partition rldlc2_gpu-l40s
##SBATCH --partition alldlc2_gpu-l40s
##SBATCH --partition alldlc2_gpu-h200
##SBATCH --partition alldlc_gpu-rtx2080
##SBATCH --partition aisdlc_gpu-rtx2080

#SBATCH --output=/work/dlclarge1/lagandua-vlawm/lumos/logs/outputs/%x_%j.out
#SBATCH --error=/work/dlclarge1/lagandua-vlawm/lumos/logs/outputs/%x_%j.err

# log basic job/cluster info
echo "== SLURM INFO =========================================================================================================="
echo "time:                     $(date)"
echo "directory:                ${PWD}"
echo "job id:                   ${SLURM_JOB_ID}"
echo "name:                     ${SLURM_JOB_NAME}"
echo "partition:                ${SLURM_JOB_PARTITION}"
echo "nodes:                    ${SLURM_NODELIST}"
echo "cpus per node:            ${SLURM_JOB_CPUS_PER_NODE}"
echo "tasks per node:           ${SLURM_TASKS_PER_NODE}"
echo "mem per node:             ${SLURM_MEM_PER_NODE}"
echo "cpus per task:            ${SLURM_CPUS_PER_TASK}"
echo "ntasks:                   ${SLURM_NTASKS}"
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
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1
export HYDRA_FULL_ERROR=1

# Job to perform
source /home/lagandua/.bashrc
conda activate $1

GPUS=$4
timeout 23.8h srun  --gres=gpu:${GPUS} python $2 slurm=true hydra.run.dir=$3 trainer.devices=$GPUS ${@:5}

if [[ $? -eq 124 ]]; then
echo "Time limit exceeded. Resubmit job.";
ssh ${USER}@kis3bat2 <<ENDSSH
sh $3/resume_training.sh
ENDSSH
fi

# Print some Information about the end-time to STDOUT
echo "DONE";
echo "Finished at $(date)";