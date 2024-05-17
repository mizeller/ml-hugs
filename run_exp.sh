# first open a screen session
screen -S hyperparams_exp

# in the screen session, run the experiment
CUDA_VISIBLE_DEVICES=4 python main.py --cfg_file cfg_files/ablation/neuman/abl_trimlp.yaml

# also run the experiment monitoring (just in case)
python ../monitor_experiments.py

echo 'leave screen session w/ ctrl+a d and reattach w/ screen -r hyperparams_exp'
