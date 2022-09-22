
python -m viskit.frontend /data/px/ceyaozhang/MyCodes/rlkit/output

nohup python -u launch_experiment.py ./configs/cheetah-dir.json --srb > cheetah-dir_sac_srb.log 2>&1 &
nohup python -u launch_experiment.py ./configs/cheetah-vel.json --srb > cheetah-vel_sac_srb.log 2>&1 &


nohup python -u launch_experiment.py ./configs/ant-dir.json --gpu_id=1 --srb > ant-dir_sac_srb.log 2>&1 &
nohup python -u launch_experiment.py ./configs/ant-goal.json --gpu_id=1 --srb > ant-goal_sac_srb.log 2>&1 &

nohup python -u launch_experiment.py ./configs/ant-dir.json --gpu_id=0 --srb > ant-dir=0_sac_srb.log 2>&1 &
nohup python -u launch_experiment.py ./configs/ant-dir.json --gpu_id=1 --srb > ant-dir=pi_sac_srb.log 2>&1 &



nohup python -u launch_experiment.py ./configs/cheetah-dir.json --gpu_id=0 --task_id 0 --srb > cheetah-dir-id0_sac_srb.log 2>&1 &
nohup python -u launch_experiment.py ./configs/cheetah-dir.json --gpu_id=1 --task_id 1 --srb > cheetah-dir-id1_sac_srb.log 2>&1 &


nohup python -u launch_experiment.py ./configs/cheetah-vel.json -alg=SAC --gpu_id=0 --task_id 0 --srb > cheetah-vel_sac_id0_srb.log 2>&1 &
nohup python -u launch_experiment.py ./configs/cheetah-vel.json -alg=SAC --gpu_id=0 --task_id 1 --srb > cheetah-vel_sac_id1_srb.log 2>&1 &
nohup python -u launch_experiment.py ./configs/cheetah-vel.json -alg=SAC --gpu_id=0 --task_id 2 --srb > cheetah-vel_sac_id2_srb.log 2>&1 &
nohup python -u launch_experiment.py ./configs/cheetah-vel.json -alg=SAC --gpu_id=1 --task_id 3 --srb > cheetah-vel_sac_id3_srb.log 2>&1 &
nohup python -u launch_experiment.py ./configs/cheetah-vel.json -alg=SAC --gpu_id=1 --task_id 4 --srb > cheetah-vel_sac_id4_srb.log 2>&1 &
nohup python -u launch_experiment.py ./configs/cheetah-vel.json -alg=SAC --gpu_id=0 --task_id 5 --srb > cheetah-vel_sac_id5_srb.log 2>&1 &
nohup python -u launch_experiment.py ./configs/cheetah-vel.json -alg=SAC --gpu_id=0 --task_id 6 --srb > cheetah-vel_sac_id6_srb.log 2>&1 &
nohup python -u launch_experiment.py ./configs/cheetah-vel.json -alg=SAC --gpu_id=0 --task_id 7 --srb > cheetah-vel_sac_id7_srb.log 2>&1 &



nohup python -u launch_experiment.py ./configs/cheetah-vel.json -alg=SAC --gpu_id=1 --task_id 62 --srb > cheetah-vel_sac_id62_srb.log 2>&1 &
nohup python -u launch_experiment.py ./configs/cheetah-vel.json -alg=SAC --gpu_id=1 --task_id 63 --srb > cheetah-vel_ssc_id63_srb.log 2>&1 &
nohup python -u launch_experiment.py ./configs/cheetah-vel.json -alg=SAC --gpu_id=0 --task_id 64 --srb > cheetah-vel_sac_id64_srb.log 2>&1 &

-----

nohup python -u launch_experiment.py ./configs/cheetah-vel.json -alg=TD3 --gpu_id=0 --task_id 0 --srb > cheetah-vel_td3_id0_srb.log 2>&1 &
nohup python -u launch_experiment.py ./configs/cheetah-vel.json -alg=TD3 --gpu_id=0 --task_id 1 --srb > cheetah-vel_td3_id1_srb.log 2>&1 &
nohup python -u launch_experiment.py ./configs/cheetah-vel.json -alg=TD3 --gpu_id=0 --task_id 2 --srb > cheetah-vel_td3_id2_srb.log 2>&1 &
nohup python -u launch_experiment.py ./configs/cheetah-vel.json -alg=TD3 --gpu_id=1 --task_id 3 --srb > cheetah-vel_td3_id3_srb.log 2>&1 &
nohup python -u launch_experiment.py ./configs/cheetah-vel.json -alg=TD3 --gpu_id=1 --task_id 4 --srb > cheetah-vel_td3_id4_srb.log 2>&1 &
nohup python -u launch_experiment.py ./configs/cheetah-vel.json -alg=TD3 --gpu_id=0 --task_id 5 --srb > cheetah-vel_td3_id5_srb.log 2>&1 &
nohup python -u launch_experiment.py ./configs/cheetah-vel.json -alg=TD3 --gpu_id=0 --task_id 6 --srb > cheetah-vel_td3_id6_srb.log 2>&1 &
nohup python -u launch_experiment.py ./configs/cheetah-vel.json -alg=TD3 --gpu_id=0 --task_id 7 --srb > cheetah-vel_td3_id7_srb.log 2>&1 &
nohup python -u launch_experiment.py ./configs/cheetah-vel.json -alg=TD3 --gpu_id=0 --task_id 8 --srb > cheetah-vel_td3_id8_srb.log 2>&1 &


nohup python -u launch_experiment.py ./configs/cheetah-vel.json -alg=TD3 --gpu_id=1 --task_id 41 --srb > cheetah-vel_td3_id41_srb.log 2>&1 &

nohup python -u launch_experiment.py ./configs/cheetah-vel.json -alg=TD3 --gpu_id=1 --task_id 42 --srb > cheetah-vel_td3_id42_srb.log 2>&1 &
nohup python -u launch_experiment.py ./configs/cheetah-vel.json -alg=TD3 --gpu_id=1 --task_id 43 --srb > cheetah-vel_td3_id43_srb.log 2>&1 &
nohup python -u launch_experiment.py ./configs/cheetah-vel.json -alg=TD3 --gpu_id=1 --task_id 44 --srb > cheetah-vel_td3_id44_srb.log 2>&1 &
nohup python -u launch_experiment.py ./configs/cheetah-vel.json -alg=TD3 --gpu_id=0 --task_id 45 --srb > cheetah-vel_td3_id45_srb.log 2>&1 &
nohup python -u launch_experiment.py ./configs/cheetah-vel.json -alg=TD3 --gpu_id=0 --task_id 46 --srb > cheetah-vel_td3_id46_srb.log 2>&1 &
nohup python -u launch_experiment.py ./configs/cheetah-vel.json -alg=TD3 --gpu_id=1 --task_id 47 --srb > cheetah-vel_td3_id47_srb.log 2>&1 &
nohup python -u launch_experiment.py ./configs/cheetah-vel.json -alg=TD3 --gpu_id=1 --task_id 48 --srb > cheetah-vel_td3_id48_srb.log 2>&1 &
nohup python -u launch_experiment.py ./configs/cheetah-vel.json -alg=TD3 --gpu_id=1 --task_id 49 --srb > cheetah-vel_td3_id49_srb.log 2>&1 &
nohup python -u launch_experiment.py ./configs/cheetah-vel.json -alg=TD3 --gpu_id=1 --task_id 50 --srb > cheetah-vel_td3_id50_srb.log 2>&1 &
nohup python -u launch_experiment.py ./configs/cheetah-vel.json -alg=TD3 --gpu_id=1 --task_id 51 --srb > cheetah-vel_td3_id51_srb.log 2>&1 &
nohup python -u launch_experiment.py ./configs/cheetah-vel.json -alg=TD3 --gpu_id=0 --task_id 52 --srb > cheetah-vel_td3_id52_srb.log 2>&1 &
nohup python -u launch_experiment.py ./configs/cheetah-vel.json -alg=TD3 --gpu_id=0 --task_id 53 --srb > cheetah-vel_td3_id53_srb.log 2>&1 &
nohup python -u launch_experiment.py ./configs/cheetah-vel.json -alg=TD3 --gpu_id=1 --task_id 54 --srb > cheetah-vel_td3_id54_srb.log 2>&1 &
nohup python -u launch_experiment.py ./configs/cheetah-vel.json -alg=TD3 --gpu_id=1 --task_id 55 --srb > cheetah-vel_td3_id55_srb.log 2>&1 &
nohup python -u launch_experiment.py ./configs/cheetah-vel.json -alg=TD3 --gpu_id=0 --task_id 56 --srb > cheetah-vel_td3_id56_srb.log 2>&1 &
nohup python -u launch_experiment.py ./configs/cheetah-vel.json -alg=TD3 --gpu_id=0 --task_id 57 --srb > cheetah-vel_td3_id57_srb.log 2>&1 &
nohup python -u launch_experiment.py ./configs/cheetah-vel.json -alg=TD3 --gpu_id=1 --task_id 58 --srb > cheetah-vel_td3_id58_srb.log 2>&1 &
nohup python -u launch_experiment.py ./configs/cheetah-vel.json -alg=TD3 --gpu_id=1 --task_id 59 --srb > cheetah-vel_td3_id59_srb.log 2>&1 &
nohup python -u launch_experiment.py ./configs/cheetah-vel.json -alg=TD3 --gpu_id=1 --task_id 60 --srb > cheetah-vel_td3_id60_srb.log 2>&1 &
nohup python -u launch_experiment.py ./configs/cheetah-vel.json -alg=TD3 --gpu_id=1 --task_id 61 --srb > cheetah-vel_td3_id61_srb.log 2>&1 &
nohup python -u launch_experiment.py ./configs/cheetah-vel.json -alg=TD3 --gpu_id=1 --task_id 62 --srb > cheetah-vel_td3_id62_srb.log 2>&1 &
nohup python -u launch_experiment.py ./configs/cheetah-vel.json -alg=TD3 --gpu_id=1 --task_id 63 --srb > cheetah-vel_td3_id63_srb.log 2>&1 &
nohup python -u launch_experiment.py ./configs/cheetah-vel.json -alg=TD3 --gpu_id=1 --task_id 64 --srb > cheetah-vel_td3_id64_srb.log 2>&1 &
