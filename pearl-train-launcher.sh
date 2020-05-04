
export N_GPUS=$(nvidia-smi -L | wc -l)
export N_JOBS=48

parallel -j $N_JOBS \
    'CUDA_VISIBLE_DEVICES=$(({%} % $N_GPUS))' python launch_pearl.py ./configs/cheetah-vel.json \
     --log_annotation restricted_train_set \
     --seed={1} \
     --num_train_steps_per_itr={2} \
    ::: 0 1 \
    ::: 1000 2000
