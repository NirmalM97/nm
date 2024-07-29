#!/bin/bash
#SBATCH --job-name=pong_dv_job
#SBATCH --output=output_%j.log
#SBATCH --error=error_%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --partition=rleap_gpu_24gb





export MINICONDA_PATH=/work/rleap1/nirmal.aheshwari/miniconda3




export CUDA_HOME=/usr/local/cuda-11.7
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/targets/x86_64-linux/lib:$LD_LIBRARY_PATH


source $MINICONDA_PATH/etc/profile.d/conda.sh

#export CC=/usr/bin/gcc
#export CXX=/usr/bin/g++
conda activate .conda/envs/SGG
#nvcc --version


#cd /work/rleap1/nirmal.aheshwari/apex
#pip uninstall -y apex
#rm -rf build/ dist/ apex.egg-info/
#python setup.py clean
#python setup.py install --cuda_ext --cpp_ext 
#--cxx11_abi --include-path $CUDA_HOME/include --cuda-gpu-archs="compute_70,sm_70,compute_75,sm_75,compute_80,sm_80"

#python -c "from apex import amp; print('Apex Installed')"

# Ensure CUDA calls are synchronized for accurate error messages
export CUDA_LAUNCH_BLOCKING=1




cd /work/rleap1/nirmal.aheshwari/Scene-Graph-Benchmark.pytorch


#rm -rf build
#python setup.py clean
#python setup.py build_ext --inplace

python3 tools/register.py --config-file configs/myconfig0.yaml


