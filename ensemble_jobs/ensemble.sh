
#PBS -S /bin/bash
#PBS -N fishnets_ensemble
#PBS -o ensemble.log
#PBS -n
#$ -M l.makinen21@imperial.ac.uk
#$ -m a

#PBS -t 0-10
#PBS -l nodes=1:has1gpu:ppn=8,walltime=24:00:00


# PBS -tc 5
#PBS -j oe	
#PBS -V



cd /data80/makinen/fishnets/ensemble_training/

echo "PBS Job Id PBS_JOBID is ${PBS_JOBID}"

echo "PBS job array index PBS_ARRAY_INDEX value is ${PBS_ARRAYID}"


# Make a subdirectory with the current PBS Job Array Index
mkdir ${PBS_ARRAYID}

module load tensorflow/2.8 
XLA_FLAGS=--xla_gpu_cuda_data_dir=\${CUDA_PATH}
export XLA_FLAGS

source /home/makinen/venvs/imnndev/bin/activate

# go to fishnets directory for models
cd /home/makinen/repositories/fishnets/

python ensemble_training.py ${PBS_ARRAYID}

# test the models on big data
#python ensemble_test.py ${PBS_ARRAYID}