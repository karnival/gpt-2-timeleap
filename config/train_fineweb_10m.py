# train a miniature character-level shakespeare model
# good for debugging and playing on macbooks and such

out_dir = 'out-fw_s10M_v1871_d256_12_const_bs16_wm5p_lr8e3'
eval_interval = 6050 # keep frequent because we'll overfit
eval_iters = 20
log_interval = 10 # don't print too too often

# we expect to overfit on this small dataset, so only save when val improves
always_save_checkpoint = True

wandb_log = True # override via command line if you like
wandb_project = 'scaling_laws'
wandb_run_name = 'fw_s10M_v1871_d256_12_const_bs16_wm5p_lr8e3'
log_activations = True

dataset = 'fineweb'
data_files = 1
gradient_accumulation_steps = 1
batch_size = 16
block_size = 2048

vocab_size = 1871

# baby GPT model :)
n_layer = 12
n_head = 4
n_embd = 256
dropout = 0

max_iters = 6050
learning_rate = 8e-3
#decay_lr = False
lr_decay='linear'
lr_decay_iters = 6050 # make equal to max_iters usually
min_lr = 8e-4

warmup_iters = 302

weight_decay = 1e-4/learning_rate
z_loss = 1e-4
