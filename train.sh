
 #PBS -S /bin/bash
 #PBS -N fishnets
 #PBS -j oe
 #PBS -o fishnets.log
 #PBS -l nodes=1:has1gpu:ppn=8,walltime=12:00:00

module load tensorflow/2.8 
XLA_FLAGS=--xla_gpu_cuda_data_dir=\${CUDA_PATH}
export XLA_FLAGS

source /home/makinen/venvs/imnndev/bin/activate

cd /home/makinen/repositories/fishnets/


python large_data.py big_model 
