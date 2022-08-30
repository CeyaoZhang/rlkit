
python -m viskit.frontend /data/px/ceyaozhang/MyCodes/rlkit/data

nohup python -u examples/sac.py ./configs/cheetah-dir.json --srb > cheetah-dir-sac-srb.log 2>&1 &
nohup python -u examples/sac.py ./configs/cheetah-vel.json --srb > cheetah-vel-sac-srb.log 2>&1 &


nohup python -u examples/sac.py ./configs/ant-dir.json --gpu_id=1 --srb > ant-dir-sac-srb.log 2>&1 &
nohup python -u examples/sac.py ./configs/ant-goal.json --gpu_id=1 --srb > ant-goal-sac-srb.log 2>&1 &