#!/bin/bash
#SBATCH --job-name=build_pytorch_env
#SBATCH --account=project_id
#SBATCH --time=00:30:00
#SBATCH --cpus-per-gpu=8
#SBATCH --gpus=1
#SBATCH --mem=30G
#SBATCH --partition=dev-g
#SBATCH -o envBuild.out

export PYTHON_ENV_NAME=env_dpo
export SING_IMAGE=lumi-pytorch-rocm-5.6.1-python-3.10-pytorch-v2.1.0.sif
cp /appl/local/containers/sif-images/lumi-pytorch-rocm-5.6.1-python-3.10-pytorch-v2.1.0.sif ./lumi-pytorch-rocm-5.6.1-python-3.10-pytorch-v2.1.0.sif 

# Create environment in the local working directory
WORKING_DIR=$(pwd)

# Build a virtual environment, activate it and install all requirements.
singularity exec -W $WORKING_DIR -B $WORKING_DIR $SING_IMAGE bash \
-c 'python3 -m venv --system-site-packages $PYTHON_ENV_NAME;. $PYTHON_ENV_NAME/bin/activate; pip install -r requirements.txt'
