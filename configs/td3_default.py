# default SAC in Meta-mujoco settings
default_config = dict(
    env_name="cheetah-dir",
    algorithm="TD3",
    version="normal",
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
    ),
    qf_kwargs=dict(
            hidden_sizes=[400, 300],
        ),
        policy_kwargs=dict(
            hidden_sizes=[400, 300],
        ),
    util_params=dict(
        use_gpu=True,
        gpu_id=0,
        seed=0,
    ),
    env_info_sizes=dict()
)
