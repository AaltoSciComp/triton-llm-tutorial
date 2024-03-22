# triton-llm-tutorial-session2
Llama2 finetuning on GPUs with jupyter notebook


### Create the conda environment
On a triton terminal, run:
```bash
mamba env create -f env2.yml -p ./myenv2
```

### Start a jupyter server on a gpu node and run jupyter notebooks on your laptop's browser

On triton:

```bash
srun --pty --mem=16G --cpus-per-task=4 --gres=gpu:1 --time=01:00:00 bash
jupyter notebook --no-browser --port=8889 --ip=0.0.0.0
```
On your laptop:
```bash
ssh -N -f -L 8889:nodename:8889 username@triton.aalto.fi
```
Then, open your browser and go to 'localhost:8889'.
