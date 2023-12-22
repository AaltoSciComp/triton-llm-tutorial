# triton-llm-tutorial-session2

On triton:
```bash
srun --pty --mem=16G --cpus-per-task=10 --gres=gpu:1 --time=01:00:00 bash
jupyter notebook --no-browser --port=8889 --ip=0.0.0.0
```
On your laptop:
```bash
ssh -N -f -L 8889:skl36:8889 tiany4@triton.aalto.fi
```



### Run the tutorial notebooks on triton
After git clone this repo to a directory on triton, you will need to define some environment variables.
Create your own `.env` file and add necessary environment variables:

```bash
#Forece transformers to use pre-downloaded models
TRANSFORMERS_OFFLINE='1'
HF_HOME='/scratch/shareddata/dldata/huggingface-hub-cache'

#optional
HUGGINGFACE_TOKEN=your_token
```
And then launch a [jupyter.triton.aalto.fi](https://jupyter.triton.aalto.fi/) server and open a notebook with the **custom kernel for this tutorial**: "LLM-tools"



---
If you need to run notebooks with your own custom kernel on [jupyter.triton.aalto.fi](https://jupyter.triton.aalto.fi/) in the future, here are the steps:
### Create a YAML File:

This file typically has a .yml or .yaml extension.

It should contain the name of the new environment and the list of packages to be installed.

Here's an example of what the content might look like:
```bash
name: myenv
channels:
  - nvidia
  - pytorch
  - conda-forge
dependencies:
  - python
  - pytorch-cuda=11.8
  - pytorch
  - sentencepiece
  - pip
  - pip:
    - transformers
    - python-dotenv
    - ipykernel
    - accelerate
```

### Create the conda environment
On a triton terminal, run:
```bash
mamba env create -f env.yml -p ./myenv
```


### Make the environment visible inside of Jupyter

```bash
module load jupyterhub/live
envkernel conda --user --name INTERNAL_NAME --display-name="My kernel" /path/to/conda_env
```

### Run the notebooks
Launch a [jupyter.triton.aalto.fi](https://jupyter.triton.aalto.fi/) server and run the notebooks with the custom kernel "My kernel"
