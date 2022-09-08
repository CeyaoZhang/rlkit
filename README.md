
python -m viskit.frontend /data/px/ceyaozhang/MyCodes/rlkit/output

nohup python -u launch_experiment.py ./configs/cheetah-dir.json --srb > cheetah-dir_sac_srb.log 2>&1 &
nohup python -u launch_experiment.py ./configs/cheetah-vel.json --srb > cheetah-vel_sac_srb.log 2>&1 &


nohup python -u launch_experiment.py ./configs/ant-dir.json --gpu_id=1 --srb > ant-dir_sac_srb.log 2>&1 &
nohup python -u launch_experiment.py ./configs/ant-goal.json --gpu_id=1 --srb > ant-goal_sac_srb.log 2>&1 &

nohup python -u launch_experiment.py ./configs/ant-dir.json --gpu_id=0 --srb > ant-dir=0_sac_srb.log 2>&1 &
nohup python -u launch_experiment.py ./configs/ant-dir.json --gpu_id=1 --srb > ant-dir=pi_sac_srb.log 2>&1 &



nohup python -u launch_experiment.py ./configs/cheetah-dir.json --gpu_id=0 --task_id 0 --srb > cheetah-dir-id0_sac_srb.log 2>&1 &
nohup python -u launch_experiment.py ./configs/cheetah-dir.json --gpu_id=1 --task_id 1 --srb > cheetah-dir-id1_sac_srb.log 2>&1 &

python launch_experiment.py ./configs/cheetah-vel.json --gpu_id=0 --task_id 0 --srb
python launch_experiment.py ./configs/cheetah-vel.json --gpu_id=1 --task_id 1 --srb
python launch_experiment.py ./configs/cheetah-vel.json --gpu_id=2 --task_id 2 --srb
python launch_experiment.py ./configs/cheetah-vel.json --gpu_id=3 --task_id 3 --srb

nohup python -u launch_experiment.py ./configs/cheetah-vel.json --gpu_id=0 --task_id 0 --srb > cheetah-vel_sac_id0_srb.log 2>&1 &
nohup python -u launch_experiment.py ./configs/cheetah-vel.json --gpu_id=0 --task_id 5 --srb > cheetah-vel_sac_id5_srb.log 2>&1 &
nohup python -u launch_experiment.py ./configs/cheetah-vel.json --gpu_id=0 --task_id 59 --srb > cheetah-vel_sac_id59_srb.log 2>&1 &
nohup python -u launch_experiment.py ./configs/cheetah-vel.json --gpu_id=0 --task_id 64 --srb > cheetah-vel_sac_id64_srb.log 2>&1 &

nohup python -u launch_experiment.py ./configs/cheetah-vel.json -alg=TD3 --gpu_id=0 --task_id 0 --srb > cheetah-vel_td3_id0_srb.log 2>&1 &
nohup python -u launch_experiment.py ./configs/cheetah-vel.json -alg=TD3 --gpu_id=0 --task_id 5 --srb > cheetah-vel_td3_id5_srb.log 2>&1 &
nohup python -u launch_experiment.py ./configs/cheetah-vel.json -alg=TD3 --gpu_id=1 --task_id 59 --srb > cheetah-vel_td3_id59_srb.log 2>&1 &
nohup python -u launch_experiment.py ./configs/cheetah-vel.json -alg=TD3 --gpu_id=1 --task_id 64 --srb > cheetah-vel_td3_id64_srb.log 2>&1 &
