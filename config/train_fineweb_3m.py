# train a miniature character-level shakespeare model
# good for debugging and playing on macbooks and such

out_dir = 'out-fw_s3M_v1539_d192_6_const_bs8_wm5p_lr8e3'
eval_interval = 3600 # keep frequent because we'll overfit
eval_iters = 20
log_interval = 10 # don't print too too often

# we expect to overfit on this small dataset, so only save when val improves
always_save_checkpoint = True

wandb_log = True # override via command line if you like
wandb_project = 'scaling_laws'
wandb_run_name = 'fw_s3M_v1539_d192_6_const_bs8_wm5p_lr8e3'
log_activations = True

dataset = 'fineweb'
data_files = 1
gradient_accumulation_steps = 1
batch_size = 8
block_size = 2048

vocab_size = 1539

# baby GPT model :)
n_layer = 6
n_head = 3
n_embd = 192
dropout = 0

max_iters = 3600
learning_rate = 8e-3
#decay_lr = False
lr_decay='linear'
lr_decay_iters = 3600 # make equal to max_iters usually
min_lr = 8e-4

warmup_iters = 180

weight_decay = 1e-4/learning_rate
z_loss = 1e-4
