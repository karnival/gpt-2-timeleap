# train a miniature character-level shakespeare model
# good for debugging and playing on macbooks and such

out_dir = 'out-s0.7M_v1168_d128_4_const_bs8_wm5p_lr8e3'
eval_interval = 1000 # keep frequent because we'll overfit
eval_iters = 20
log_interval = 10 # don't print too too often

# we expect to overfit on this small dataset, so only save when val improves
always_save_checkpoint = True

wandb_log = True # override via command line if you like
wandb_project = 'scaling_laws'
wandb_run_name = 's0.7M_v1168_d128_4_const_bs8_wm5p_lr8e3'

dataset = 'fineweb'
data_files = 1
gradient_accumulation_steps = 1
batch_size = 8
block_size = 2048

vocab_size = 2000

# baby GPT model :)
n_layer = 4
n_head = 2
n_embd = 128
dropout = 0

max_iters = 1000
learning_rate = 8e-3
decay_lr = False
#lr_decay='linear'
#lr_decay_iters = 1000 # make equal to max_iters usually
min_lr = 8e-4

warmup_iters = 50

weight_decay = 1e-4/learning_rate
z_loss = 1e-4
