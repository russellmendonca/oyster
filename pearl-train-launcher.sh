
export N_GPUS=$(nvidia-smi -L | wc -l)
export N_JOBS=48

parallel -j $N_JOBS \
    'CUDA_VISIBLE_DEVICES=$(({%} % $N_GPUS))' python launch_pearl.py ./configs/envs/ant-crippled.json 2000-steps \
     --seed={1} \
    ::: 0 1 2
