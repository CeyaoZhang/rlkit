
python -m viskit.frontend /data/px/ceyaozhang/MyCodes/rlkit/output

nohup python -u examples/sac.py ./configs/cheetah-dir.json --srb > cheetah-dir-sac-srb.log 2>&1 &
nohup python -u examples/sac.py ./configs/cheetah-vel.json --srb > cheetah-vel-sac-srb.log 2>&1 &


nohup python -u examples/sac.py ./configs/ant-dir.json --gpu_id=1 --srb > ant-dir-sac-srb.log 2>&1 &
nohup python -u examples/sac.py ./configs/ant-goal.json --gpu_id=1 --srb > ant-goal-sac-srb.log 2>&1 &

nohup python -u examples/sac.py ./configs/ant-dir.json --gpu_id=0 --srb > ant-dir=0-sac-srb.log 2>&1 &
nohup python -u examples/sac.py ./configs/ant-dir.json --gpu_id=1 --srb > ant-dir=pi-sac-srb.log 2>&1 &


nohup python -u examples/sac.py ./configs/cheetah-vel.json --task_id 0 --srb > cheetah-vel-id0-sac-srb.log 2>&1 &