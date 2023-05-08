graph_size=1000
knapsack_alg=ppo # algorithm
knapsack_enc=GATEncoder
#knapsack_enc=GraphAttentionEncoderMask
run_name=cvrp"$graph_size"_"$knapsack_alg"_"$knapsack_enc"
lr_knapsack=1e-4 # outer agent policy
lr_model=0 # inner agent
device=cuda:1
load_path=pretrained/good_buffer_cvrp_baseline_20230130T121646/epoch-99.pt
loss_weights="1 1 0"

python run.py --problem cvrp --graph_size ${graph_size} --baseline rollout --run_name ${run_name} \
   --n_epochs 100 --epoch_size 76800 --batch_size 32 --eval_batch_size 32 --no_tensorboard \
   --lr_model ${lr_model} --lr_knapsack ${lr_knapsack} \
   --load_path ${load_path} --device ${device} --knapsack_alg ${knapsack_alg} \
   --loss_weights ${loss_weights} --checkpoint_epochs 1 \
   --symmetric_force --knapsack_enc ${knapsack_enc}