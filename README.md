nohup python -u examples/sac.py cheetah-dir --srb > cheetah-dir-sac-srb.log 2>&1 &
nohup python -u examples/sac.py cheetah-vel --srb > cheetah-vel-sac-srb.log 2>&1 &


nohup python -u examples/sac.py ant-dir --gpu_id=1 --srb > ant-dir-sac-srb.log 2>&1 &
nohup python -u examples/sac.py ant-goal --gpu_id=1 --srb > ant-goal-sac-srb.log 2>&1 &