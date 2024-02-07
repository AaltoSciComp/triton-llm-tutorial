#!/bin/bash
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
# Load conda and the extra packages that were not installed in the container.
#
\$WITH_CONDA
export PYTHONPATH=/workdir/env5.6/lib/python3.10/site-packages
export MIOPEN_USER_DB_PATH="/tmp/$(whoami)-miopen-cache-\$SLURM_NODEID"
export MIOPEN_CUSTOM_CACHE_DIR=\$MIOPEN_USER_DB_PATH
export HF_HOME=/workdir/
export TRANSFORMERS_OFFLINE=0
export HF_DATASETS_OFFLINE=0
set -x

python /workdir/hfrl_dpo_gpu_lumi.py
EOF
chmod +x $wd/run-me.sh
#
# Set snapshot dir and use no workers.
#


cd $wd
rm -rf $wd/rank-*.log $wd/gpt_snapshot.pt

singularity exec \
        -B "$wd:/workdir" \
         $SIF /workdir/run-me.sh
