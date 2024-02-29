## PPO examples on triton

### Create the conda environment
On a triton terminal, run:
```bash
mamba env create -f env.yml -p ./myenv
```

### Make the environment visible (a customized kernel) inside of Jupyter

```bash
module load jupyterhub/live

envkernel conda --user --name INTERNAL_NAME --display-name="My kernel" /path/to/myenv
```
### For the cpu example, you can use jupyter triton directly.

### For the gpu example, first start a jupyter server on a gpu node and then run jupyter notebooks on your laptop's browser

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
Then, open your browser and go to 'localhost:8889', choose python3 kernel, instead of using the customized kernel.
