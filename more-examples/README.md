## [hfrl_ppo_gpu_lumi.ipynb](hfrl_ppo_gpu_lumi.ipynb)
### Get an interactive job
On a lumi terminal, run:
```bash
srun --account=project_id --time=02:00:00 --partition=standard-g --nodes=1 --mem-per-cpu=8G --ntasks=1 --cpus-per-task=10 --gpus-per-task=2 --pty bash
```
After resouces allocated, start the container in a interactive manner:
```bash
singularity exec -B ./:/workdir pytorch_rocm5.7_ubuntu22.04_py3.10_pytorch_2.0.1.sif bash
```
Creat a python virtual environment and active it: 
```bash
python -m venv myenv --system-site-packages
source myenv/bin/activate
```
Install extra packages:
```bash
pip install -r requirments.txt
```

In the container, start a notebook server 
```bash
python -m notebook --no-browser --port=8889 --ip=0.0.0.0
```
On your laptop:
```bash
ssh -N -f -L 8889:nodename:8889 username@lumi.csc.fi
```
Then, open your browser and go to 'localhost:8889'
