from gym.envs.mujoco import HalfCheetahEnv

import rlkit.torch.pytorch_util as ptu
from rlkit.data_management.env_replay_buffer import EnvReplayBuffer
from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.launchers.launcher_util import setup_logger, set_seed
from rlkit.samplers.data_collector import MdpPathCollector
from rlkit.torch.sac.policies import TanhGaussianPolicy, MakeDeterministic
from rlkit.torch.sac.sac import SACTrainer
from rlkit.torch.networks import ConcatMlp
from rlkit.torch.torch_rl_algorithm import TorchBatchRLAlgorithm

from rlkit.envs.pearl_envs import ENVS
from configs.default import default_config

import os, json
import numpy as np

def deep_update_dict(fr, to):
    ''' update dict of dicts with new values '''
    # assume dicts have same keys
    for k, v in fr.items():
        if type(v) is dict:
            deep_update_dict(v, to[k])
        else:
            to[k] = v
    return to

def experiment(variant):

    # expl_env = NormalizedBoxEnv(HalfCheetahEnv())
    # eval_env = NormalizedBoxEnv(HalfCheetahEnv())
    # obs_dim = expl_env.observation_space.low.size
    # action_dim = eval_env.action_space.low.size
    
    # print(ENVS)
    ## to-do-list: we need to set if-else here under different env
    env = NormalizedBoxEnv(ENVS[variant['env_name']](n_tasks=variant['env_params']["n_tasks"])) ## ENVS[variant['env_name']]() is an object of env
    task_id = variant['task_id']
    assert task_id < variant['env_params']["n_tasks"], "the task id should less than the num tasks in this env"
    
    env.reset_task(task_id)

    obs_dim = int(np.prod(env.observation_space.shape))
    action_dim = int(np.prod(env.action_space.shape))
    eval_env = env
    expl_env = env

    M = variant['layer_size']
    qf1 = ConcatMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M],
    )
    qf2 = ConcatMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M],
    )
    target_qf1 = ConcatMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M],
    )
    target_qf2 = ConcatMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M],
    )

    policy = TanhGaussianPolicy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_sizes=[M, M],
    )
    eval_policy = MakeDeterministic(policy) ## which make eval performance is better than expl performance

    eval_path_collector = MdpPathCollector(
        eval_env,
        eval_policy,
    )
    expl_path_collector = MdpPathCollector(
        expl_env,
        policy,
    )
    
    replay_buffer = EnvReplayBuffer(
        max_replay_buffer_size=variant['replay_buffer_size'],
        env=expl_env,
        env_info_sizes=variant['env_info_sizes']
    )

    trainer = SACTrainer(
        env=eval_env,
        policy=policy,
        qf1=qf1,
        qf2=qf2,
        target_qf1=target_qf1,
        target_qf2=target_qf2,
        **variant['trainer_kwargs']
    )

    algorithm = TorchBatchRLAlgorithm(
        trainer=trainer,
        exploration_env=expl_env,
        evaluation_env=eval_env,
        exploration_data_collector=expl_path_collector,
        evaluation_data_collector=eval_path_collector,
        replay_buffer=replay_buffer,
        **variant['algorithm_kwargs']
    )
    algorithm.to(ptu.device)
    algorithm.train()


import click
@click.command()
@click.argument('config', default="./configs/cheetah-dir.json")
@click.option('--task_id', type=int, default=0) ## you can choose the task id in each env, but we have some constrains in some env
@click.option('--seed', type=int, default=100) 
@click.option('--use_gpu/--use_cpu', default=True)
@click.option('--gpu_id', default=0)
@click.option('--uaet/--nuaet', is_flag=True, default=False) # default not use_automatic_entropy_tuning
@click.option('--srb/--nsrb', is_flag=True, default=False) # save replay buffer
def main(config, task_id, seed, use_gpu, gpu_id, srb, uaet): 

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
    variant['trainer_kwargs']['use_automatic_entropy_tuning'] = uaet

    set_seed(seed)
    ptu.set_gpu_mode(mode=use_gpu, gpu_id=gpu_id)  # optionally set the GPU (default=True)

    setup_logger(exp_prefix=variant['env_name'], variant=variant, snapshot_mode="last")

    experiment(variant)



if __name__ == "__main__":
    main()

