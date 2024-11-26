dataset = 'fw'

n = 1
beta2 = [0.95]
lr = 5e-3
bs = 8
ga = 1
max_overtrain = 8

model = {
    "n_layer": 4,
    "n_head": 2,
    "n_embd": 128,
    "vocab_size": 1168,
    "max_tokens": ((12*4*128**2 + 128*1168)*20*max_overtrain // (2048*bs*ga)) * (2048*bs*ga),
    "dataset": f"{dataset}_1168",
    "d_files": 1,
        },


n_iters = int(model['max_tokens']) // (2048*bs*ga)
eval_interval = n_iters//(10*max_overtrain//4)
warmup = n_iters // (20*max_overtrain)

for i in range(0, max_overtrain*20, 8):
    if i == 0:
        name = 'backbone1'
        init_from = 'scracth_0'
        max_iters = n_iters
        decay_lr = False
    else:
        name = f'm{str(i)}'
        init_from = 'resume_0'
        max_iters = eval_interval * i
        decay_lr = True
    config = f"""
out_dir = 'out-lrtest_{name}_fw_s{n}M_v{params['vocab_size']}_d{params['n_embd']}_l{params['n_layer']}_lin_bs{bs*ga}_wm5p_lr{lr}_b2{b2}'
eval_interval = {eval_interval} # keep frequent because we'll overfit
eval_iters = 200
log_interval = 10 # don't print too too often

always_save_checkpoint = True

wandb_log = True # override via command line if you like
wandb_project = 'scaling_laws'
wandb_run_name = 'lrtest_backbone1_fw_s{n}M_v{params['vocab_size']}_d{params['n_embd']}_l{params['n_layer']}_lin_bs{bs*ga}_wm5p_lr{lr}_b2{b2}'
log_activations = False
init_from = {init_from}

dataset = "{params['dataset']}"
data_files = {params['d_files']}
gradient_accumulation_steps = {ga}
batch_size = {bs}
block_size = 2048

vocab_size = {params['vocab_size']}

# baby GPT model :)
n_layer = {params['n_layer']}
n_head = {params['n_head']}
n_embd = {params['n_embd']}
dropout = 0

max_iters = {max_iters}
learning_rate = {lr}
decay_lr = {decay_lr}
lr_decay='linear'
lr_decay_iters = {max_iters} # make equal to max_iters usually
min_lr = {lr/10}

beta2 = {b2}

warmup_iters = {warmup}

weight_decay = 1e-4/learning_rate
z_loss = 1e-4
                """

    with open(f"train_{name}.py", "w") as f:
        f.write(config)
