#!/bin/bash


#SBATCH -J test_run
#name of job

#SBATCH -o output.out
#standard output file

#SBATCH --nodelist=dscog027

#SBATCH --gres=gpu:1

#SBATCH -n 1
#number of tasks

#SBATCH -p gpu_7day
#queue for the job

#SBATCH -t 8:00:00

. $HOME/bashrc

conda init
conda activate mesaverde

nvidia-smi

echo ""
echo "<--- Job output --->"
echo ""

# Put job commands here V

python forward_prediction.py ~/3DPBE/outforward/config.json ~/3DPBE/id_prop/POSCAR__jsonNum_0_entryNum100000_agm003704385.vasp

# Put job commands above ^

echo ""
echo "{JOB FINISHED}"
echo ""
echo "Job ID:              $SLURM_JOB_ID"
echo "Job Name:            $SLURM_JOB_NAME"
echo "Number of Nodes:     $SLURM_JOB_NUM_NODES"
echo "Number of CPU cores: $SLURM_CPUS_ON_NODE"
echo "Number of Tasks:     $SLURM_NTASKS"
echo "Partition:           $SLURM_JOB_PARTITION"
