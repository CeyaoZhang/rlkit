# default SAC in Meta-mujoco settings
default_config = dict(
    env_name="cheetah-dir",
    algorithm="SAC",
    version="normal",
    layer_size=256, 
    task_id=0,
    env_params=dict(
        n_tasks=2,
    ),
    replay_buffer_size=int(3E6), # default is 1e6
    algorithm_kwargs=dict(
        batch_size=256,
        max_path_length=200, # default is 1000
        num_epochs=3000,
        num_eval_steps_per_epoch=600,
        num_expl_steps_per_train_loop=1000,
        num_trains_per_train_loop=2000,
        min_num_steps_before_training=2000,
        save_replay_buffer=False,
    ),
    trainer_kwargs=dict(
        discount=0.99,
        soft_target_tau=5e-3,
        target_update_period=1,
        policy_lr=3E-4,
        qf_lr=3E-4,
        reward_scale=1,
        use_automatic_entropy_tuning=False,
    ),
    util_params=dict(
        use_gpu=True,
        gpu_id=0,
        seed=0,
    ),
    env_info_sizes=dict()
)
