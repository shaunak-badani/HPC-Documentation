#!/bin/zsh
#SBATCH --mem=16G # 16 GBs RAM 
#SBATCH -p courses-gpu 
#SBATCH --gres=gpu:2 
#SBATCH --exclusive


MY_NET_ID=skb79
LARGE_HOME="/hpc/group/coursess25/aipi590/${MY_NET_ID}"

source $LARGE_HOME/.venv/bin/activate

python test_script.py