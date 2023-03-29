run_name=cvrp20
knapsack_alg=ppo # algorithm
lr_knapsack=1e-4 # outer agent policy
lr_model=1e-5 # inner agent
device=cuda:1
load_path=pretrained/good_buffer_cvrp_baseline_20230130T121646/epoch-99.pt
loss_weights="1 1 0.1"

python run.py --problem cvrp --graph_size 20 --baseline rollout --run_name ${run_name} \
   --n_epochs 100 --epoch_size 76800 --batch_size 512 --no_tensorboard \
   --lr_model ${lr_model} --lr_knapsack ${lr_knapsack} \
   --load_path ${load_path} --device ${device} --knapsack_alg ${knapsack_alg} \
   --loss_weights ${loss_weights} \
   --symmetric_force