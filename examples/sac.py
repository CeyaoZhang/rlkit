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

import numpy as np


def experiment(variant):

    # expl_env = NormalizedBoxEnv(HalfCheetahEnv())
    # eval_env = NormalizedBoxEnv(HalfCheetahEnv())
    # obs_dim = expl_env.observation_space.low.size
    # action_dim = eval_env.action_space.low.size

    # from rlkit.envs.pearl_envs.half_cheetah_dir import HalfCheetahDirEnv
    # expl_env = NormalizedBoxEnv(HalfCheetahDirEnv())
    # eval_env = NormalizedBoxEnv(HalfCheetahDirEnv())
    
    # print(ENVS)
    env = NormalizedBoxEnv(ENVS[variant['env_name']]()) ## ENVS[variant['env_name']]() is an object of env
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
    eval_policy = MakeDeterministic(policy)
    eval_path_collector = MdpPathCollector(
        eval_env,
        eval_policy,
    )
    expl_path_collector = MdpPathCollector(
        expl_env,
        policy,
    )
    replay_buffer = EnvReplayBuffer(
        variant['replay_buffer_size'],
        expl_env,
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
@click.argument('env_name', default='cheetah-dir')
@click.option('--seed', type=int, default=100) 
@click.option('--use_gpu/--use_cpu', default=True)
@click.option('--gpu_id', default=0)
@click.option('--uaet/--nuaet', is_flag=True, default=False) # default not use_automatic_entropy_tuning
@click.option('--srb/--nsrb', is_flag=True, default=False) # save replay buffer
def main(env_name, seed, use_gpu, gpu_id, uaet, srb): 

    set_seed(seed)
    ptu.set_gpu_mode(mode=use_gpu,gpu_id=gpu_id)  # optionally set the GPU (default=False)

    # noinspection PyTypeChecker
    variant = dict(
        algorithm="SAC",
        version="normal",
        layer_size=256,
        replay_buffer_size=int(3E6), # default is 1e6
        algorithm_kwargs=dict(
            num_epochs=3000,
            num_eval_steps_per_epoch=5000,
            num_trains_per_train_loop=1000,
            num_expl_steps_per_train_loop=1000,
            min_num_steps_before_training=1000,
            max_path_length=1000,
            batch_size=256,
            save_replay_buffer=srb,
        ),
        trainer_kwargs=dict(
            discount=0.99,
            soft_target_tau=5e-3,
            target_update_period=1,
            policy_lr=3E-4,
            qf_lr=3E-4,
            reward_scale=1,
            use_automatic_entropy_tuning=uaet,
        ),
        env_name=env_name
    )

    setup_logger(exp_prefix=env_name, variant=variant)
    experiment(variant)



if __name__ == "__main__":
    main()
    # # noinspection PyTypeChecker
    # variant = dict(
    #     algorithm="SAC",
    #     version="normal",
    #     layer_size=256,
    #     replay_buffer_size=int(1E6), 
    #     algorithm_kwargs=dict(
    #         num_epochs=3000,
    #         num_eval_steps_per_epoch=5000,
    #         num_trains_per_train_loop=1000,
    #         num_expl_steps_per_train_loop=1000,
    #         min_num_steps_before_training=1000,
    #         max_path_length=1000,
    #         batch_size=256,
    #     ),
    #     trainer_kwargs=dict(
    #         discount=0.99,
    #         soft_target_tau=5e-3,
    #         target_update_period=1,
    #         policy_lr=3E-4,
    #         qf_lr=3E-4,
    #         reward_scale=1,
    #         use_automatic_entropy_tuning=True,
    #     ),
    # )
    # setup_logger('name-of-experiment', variant=variant)
    # ptu.set_gpu_mode(True)  # optionally set the GPU (default=False)
    # experiment(variant)
