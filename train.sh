#!/bin/zsh
#SBATCH --mem=8G # 8 GBs RAM 
#SBATCH -p courses-gpu 
#SBATCH --gres=gpu:1
#SBATCH --exclusive


MY_NET_ID=skb79
LARGE_HOME="/hpc/group/coursess25/aipi590/${MY_NET_ID}"

source $LARGE_HOME/.venv/bin/activate

python test_script.py