# train a miniature character-level shakespeare model
# good for debugging and playing on macbooks and such

out_dir = 'out-fineweb'
eval_interval = 200 # keep frequent because we'll overfit
eval_iters = 200
log_interval = 10 # don't print too too often

# we expect to overfit on this small dataset, so only save when val improves
always_save_checkpoint = True

wandb_log = True # override via command line if you like
wandb_project = 'scaling_laws'
wandb_run_name = 's5M_v2000_d256_l6_const_bs40_wm5p_lr8e3'

dataset = 'fineweb'
gradient_accumulation_steps = 1
batch_size = 20*2
block_size = 1024

vocab_size = 2000

# baby GPT model :)
n_layer = 6
n_head = 4
n_embd = 256
dropout = 0

max_iters = 10000
learning_rate = 8e-3
lr_decay='linear'
lr_decay_iters = 20000 # make equal to max_iters usually
min_lr = 8e-4

warmup_iters = 130

weight_decay=1e-4/learning_rate
