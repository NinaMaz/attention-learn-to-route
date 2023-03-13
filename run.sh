run_name=cvrp20_ac_fixed_masked_selected
knapsack_agent=actor-critic
lr_knapsack=1e-4
lr_model=1e-5
device=cuda:1
load_path=pretrained/good_buffer_cvrp_baseline_20230130T121646/epoch-99.pt
python run.py --problem cvrp --graph_size 20 --baseline rollout --n_epochs 100 \
   --epoch_size 76800 --batch_size 512 --no_tensorboard \
   --run_name ${run_name} --lr_model ${lr_model} --lr_knapsack ${lr_knapsack} \
   --load_path ${load_path} --device ${device} --knapsack_agent ${knapsack_agent} \
#   --symmetric_force