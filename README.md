# triton-llm-tutorial
This is the repo for LLMs-on-triton tutorials.

---
Here are the steps to set up environment and run notebooks on [jupyter.triton.aalto.fi](https://jupyter.triton.aalto.fi/):

### Create a conda environment
From terminal, run:
```bash
conda env create -f env.yml -p ./myenv
```


### Make the environment visible inside of Jupyter

For conda environments, you can do:
```bash
module load jupyterhub/live
envkernel conda --user --name INTERNAL_NAME --display-name="My conda" /path/to/conda_env
```

Create your own `.env` file and add necessary environment variables, for example:

```bash
TRANSFORMERS_OFFLINE='1'
HF_HOME='/scratch/shareddata/dldata/huggingface-hub-cache'
HUGGINGFACE_TOKEN=your_token
```

### Launch a [jupyter.triton.aalto.fi](https://jupyter.triton.aalto.fi/) server to run the notebooks.
