# train a miniature character-level shakespeare model
# good for debugging and playing on macbooks and such

out_dir = 'out-fineweb'
eval_interval = 250 # keep frequent because we'll overfit
eval_iters = 200
log_interval = 10 # don't print too too often

# we expect to overfit on this small dataset, so only save when val improves
always_save_checkpoint = False

wandb_log = True # override via command line if you like
wandb_project = 'fineweb'
wandb_run_name = 'improved_gpt_124m'

dataset = 'fineweb'
gradient_accumulation_steps = 5*8
batch_size = 12
block_size = 1024

# baby GPT model :)
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0

max_iters = 5000
lr_decay_iters = 5000 # make equal to max_iters usually

warmup_iters = 250
