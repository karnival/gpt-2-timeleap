# train a miniature character-level shakespeare model
# good for debugging and playing on macbooks and such

out_dir = 'out-s16M_v5000_d320_l12_const_bs90_wm5p_lr6e3'
eval_interval = 200 # keep frequent because we'll overfit
eval_iters = 200
log_interval = 10 # don't print too too often

# we expect to overfit on this small dataset, so only save when val improves
always_save_checkpoint = True

wandb_log = True # override via command line if you like
wandb_project = 'scaling_laws'
wandb_run_name = 's16M_v5000_d320_l12_const_bs90_wm5p_lr6e3'

dataset = 'fineweb'
gradient_accumulation_steps = 1
batch_size = 20*2
block_size = 1024

vocab_size = 5000

# baby GPT model :)
n_layer = 12
n_head = 5
n_embd = 320
dropout = 0

max_iters = 14000
learning_rate = 6e-3
lr_decay='linear'
lr_decay_iters = 28000 # make equal to max_iters usually
min_lr = 6e-4

warmup_iters = 175

weight_decay=1e-4/learning_rate
