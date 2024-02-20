#!/bin/bash
#SBATCH --job-name=dpoExample
#SBATCH --account=project_id
#SBATCH --time=01:00:00
#SBATCH --exclusive
#SBATCH --nodes=1
#SBATCH --cpus-per-gpu=7
#SBATCH --gpus-per-node=8
#SBATCH --mem=480G
#SBATCH --partition=standard-g
#SBATCH -o dpo.out

wd=$(realpath .)
SIF=lumi-pytorch-rocm-5.6.1-python-3.10-pytorch-v2.1.0.sif

rm -rf $wd/run-me.sh
cat > $wd/run-me.sh << EOF
#!/bin/bash -e
echo "Rank \$SLURM_PROCID - \$(taskset -p \$\$) \$ROCR_VISIBLE_DEVICES"
# Make sure GPUs are up, this seems to sometimes be necessary on lumi... 
if [ \$SLURM_LOCALID -eq 0 ] ; then
    rocm-smi 
fi
sleep 2

#
# Load conda and the extra packages that are not in the container.
#
\$WITH_CONDA
export PYTHONPATH=/workdir/env_dpo/lib/python3.10/site-packages
# export MIOPEN_USER_DB_PATH="/tmp/$(whoami)-miopen-cache-\$SLURM_NODEID"
# export MIOPEN_CUSTOM_CACHE_DIR=\$MIOPEN_USER_DB_PATH

# use the project directory as Huggingface cache folder
export HF_HOME=/workdir/
export HF_TOKEN=yourtoken

# use cached/predownloaded model weights and datasets
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

set -x
python /workdir/rlhf_dpo_multi_gpu.py
EOF
chmod +x $wd/run-me.sh

#
singularity exec \
        -B "$wd:/workdir" \
         $SIF /workdir/run-me.sh
