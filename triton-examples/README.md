## PPO examples on triton

### Create the conda environment

On a triton terminal, run:
```bash
mamba env create -f env.yml -p ./myenv
```
### Start a jupyter server on a gpu/cpu node and then run jupyter notebooks on your laptop's browser

On triton:

```bash
srun --pty --mem=32G --cpus-per-task=4 --gres=gpu:1 --time=02:00:00 bash
conda activate ./myenv
jupyter notebook --no-browser --port=8889 --ip=0.0.0.0
```
On your laptop:
```bash
ssh -N -f -L 8889:nodename:8889 username@triton.aalto.fi
```
Then, open your browser and go to 'localhost:8889', choose python3 kernel.
