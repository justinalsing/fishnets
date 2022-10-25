
#!/bin/bash
#SBATCH --account=rrg-lplevass
#SBATCH --gres=gpu:1       # Request GPU "generic resources"
#SBATCH --cpus-per-task=6  # Cores proportional to GPUs: 6 on Cedar, 10 on Béluga, 16 on Graham.
#SBATCH --mem=64G       # Memory proportional to GPUs: 32000 Cedar, 47000 Béluga, 64000 Graham.
#SBATCH --time=0-010:00     # DD-HH:MM:SS

source /home/makinen/venvs/imnndev/bin/activate

python main.py --runner AnnealRunner --config quijote.yml --doc quijote
