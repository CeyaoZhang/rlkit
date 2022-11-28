import os
import json

import rlkit.torch.pytorch_util as ptu
from rlkit.launchers.launcher_util import setup_logger, set_seed

def deep_update_dict(fr, to):
    ''' update dict of dicts with new values '''
    # assume dicts have same keys
    for k, v in fr.items():
        if type(v) is dict:
            deep_update_dict(v, to[k])
        else:
            to[k] = v
    return to

import click
@click.command()
@click.argument('config', default="./configs/cheetah-dir.json")
@click.option('--algorithm', '-alg', default="SAC")
@click.option('--task_id', type=int, default=0) ## you can choose the task id in each env, but we have some constrains in some env
@click.option('--seed', type=int, default=0) 
@click.option('--use_gpu/--use_cpu', default=True)
@click.option('--gpu_id', default=0)
# @click.option('--uaet/--nuaet', is_flag=True, default=False) # default not use_automatic_entropy_tuning
@click.option('--srb/--nsrb', is_flag=True, default=False) # save replay buffer
@click.option('--spe', type=int, default=100) # save replay buffer per epochs, the new one will cover the previous one
def main(config, algorithm, task_id, seed, use_gpu, gpu_id, srb, spe): 

    if algorithm == "SAC":
        from configs.sac_default import default_config
        from examples.sac import experiment
    elif algorithm == "TD3":
        from configs.td3_default import default_config
        from examples.td3 import experiment

    variant = default_config
    if config:
        with open(os.path.join(config)) as f:
            exp_params = json.load(f)
        variant = deep_update_dict(exp_params, variant)
    
    variant['task_id'] = task_id
    variant['util_params']['seed'] = seed
    variant['util_params']['use_gpu'] = use_gpu
    variant['util_params']['gpu_id'] = gpu_id
    
    variant['algorithm_kwargs']['save_replay_buffer'] = srb 
    variant['algorithm_kwargs']['save_per_epoch'] = spe 
    # variant['trainer_kwargs']['use_automatic_entropy_tuning'] = uaet

    set_seed(seed)
    ptu.set_gpu_mode(mode=use_gpu, gpu_id=gpu_id)  # optionally set the GPU (default=True)

    setup_logger(exp_prefix=variant['env_name'], variant=variant, snapshot_mode="last")

    experiment(variant)



if __name__ == "__main__":
    main()