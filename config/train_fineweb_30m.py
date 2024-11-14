# train a miniature character-level shakespeare model
# good for debugging and playing on macbooks and such

out_dir = 'out-fw_s30M_v2466_d384_18_const_bs64_wm5p_lr8e3'
eval_interval = 5000 # keep frequent because we'll overfit
eval_iters = 20
log_interval = 10 # don't print too too often

# we expect to overfit on this small dataset, so only save when val improves
always_save_checkpoint = True

wandb_log = True # override via command line if you like
wandb_project = 'scaling_laws'
wandb_run_name = 'fw_s30M_v2466_d384_18_const_bs64_wm5p_lr8e3'
log_activations = True

dataset = 'fineweb'
data_files = 1
gradient_accumulation_steps = 1
batch_size = 16
block_size = 2048

vocab_size = 2466

# baby GPT model :)
n_layer = 18
n_head = 6
n_embd = 384
dropout = 0

max_iters = 5000
learning_rate = 2e-3
#decay_lr = False
lr_decay='linear'
lr_decay_iters = 5000 # make equal to max_iters usually
min_lr = 2e-4

warmup_iters = 250

weight_decay = 1e-4/learning_rate
z_loss = 1e-4
