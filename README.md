# HPC Guide

> A guide to setting up Duke Compute clusters.


#### Steps

1. Create an ssh public private key pair, and add your public key to the duke portal. See setup instructions [here](https://vcm.duke.edu/help/23).

2. Add the following to your `~/.ssh/config` file (Create if not present):

```
Host dcc
HostName dcc-login.oit.duke.edu
User <net-id>
IdentityFile <path-to-private key>
```

For example, my config looks like so:

```
Host dcc
HostName dcc-login.oit.duke.edu
User skb79
IdentityFile ~/.ssh/keys/id_ed25519
```

3. Try ssh-ing into the duke server using the command: `ssh dcc`. If everything is setup correctly, you should see a prompt like so:

```
[skb79@dcc-login-04]~%
```

4. Create your directory in the aipi590 group:
```
MY_NET_ID=<your-net-id>
LARGE_HOME="/hpc/group/coursess25/aipi590/${MY_NET_ID}"
mkdir -p $LARGE_HOME
```

This folder has a lot of memory that you can use, to store your models, or config files, or anything extra. Try to reserve space in your home directory as much as possible (`/hpc/home/skb79`)

5. Create virtual environment and activate it:

```
python -m venv $LARGE_HOME/.venv
source $LARGE_HOME/.venv/bin/activate
```

From now on, all commands that you do will require some extra memory support, so request an interactive session with 8 GB of RAM and 2 CPUs, running on the zsh shell:

```
srun -p courses --mem=8G --account=coursess25 -c 2 --pty zsh -i
```


6. Install torch with cuda support:
```
pip install torch==2.1.0 torchvision -f https://download.pytorch.org/whl/cu118/torch_stable.html
```

7. Install additional requirements:
```
pip install -r requirements.txt
```

8. With this, if you run the script `python test_script.py`, you will find that the script is using cpu. This is because we initially requested cpus in our interactive session. Exit this session.

9. Submit the job script in `train.sh` using the following command:

```
sbatch train.sh
```
This command submits your job in a queue. You can check the list of jobs you have submitted using the following command:

```
squeue -u $USER
```

Sample output:
```
27137647 courses-g train.sh    skb79 PD       0:00      1 (Priority)
```
(Priority) means that the job is in queue, (Resources) mean

While your job is running, the output of the job is generated in a file called `slurm-<job-id>.out`. You can check the output of your script so far using the command:

```
cat slurm*.out 
```

You can even ssh into the node that you obtain, and see GPU usage:

```
ssh dcc-courses-gpu-<gpu-number-allotted>
nvidia-smi
```

You can exit out of dcc by typing:
```
exit
```