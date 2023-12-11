# triton-llm-tutorial
This is the repo for LLMs-on-triton tutorials.

---
Here are the steps to set up a conda environment and run notebooks on [jupyter.triton.aalto.fi](https://jupyter.triton.aalto.fi/):

### Create a conda environment
On a triton terminal, run:
```bash
mamba env create -f env.yml -p ./myenv
```


### Make the environment visible inside of Jupyter

```bash
module load jupyterhub/live
envkernel conda --user --name INTERNAL_NAME --display-name="My conda" /path/to/conda_env
```
### Define some environment variables
Create your own `.env` file and add necessary environment variables, for example:

```bash
TRANSFORMERS_OFFLINE='1'
HF_HOME='/scratch/shareddata/dldata/huggingface-hub-cache'
HUGGINGFACE_TOKEN=your_token
```
### Run the notebooks
Launch a [jupyter.triton.aalto.fi](https://jupyter.triton.aalto.fi/) server and run the notebooks with the custom kernel "My conda"
