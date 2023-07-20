
load_path=/trinity/home/n.mazyavkina/attention-learn-to-route/pretrained/tsp_20

python run.py --problem cvrp --graph_size 20 --baseline rollout --run_name 'tam_vrp_20' --fixed_logist ${load_path} --no_tensorboard