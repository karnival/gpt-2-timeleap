import os

dataset = 'fineweb'

version = 1

n = 3
b2 = 0.95
ga = 1
max_overtrain = 8

models =  {
    1: {
    "n_layer": 4,
    "n_head": 2,
    "n_embd": 128,
    "vocab_size": 1168,
    "max_tokens": ((12*4*128**2 + 128*1168)*20*max_overtrain // (2048*8)) * (2048*8),
    "dataset": f"{dataset}_1168",
    "d_files": 1,
    "batch_size": 8,
    "learning_rate": 5e-3
        },
    3: {
    "n_layer": 6,
    "n_head": 3,
    "n_embd": 192,
    "vocab_size": 1539,
    "max_tokens": ((12*6*192**2 + 192*1539)*20*max_overtrain // (2048*16)) * (2048*16),
    "dataset": f"{dataset}_1539",
    "d_files": 1,
    "batch_size": 16,
    "learning_rate": 2e-3
        },
    10: {
    "n_layer": 12,
    "n_head": 4,
    "n_embd": 256,
    "vocab_size": 1871,
    "max_tokens": ((12*12*256**2 + 256*1871)*20*max_overtrain // (2048*32)) * (2048*32),
    "dataset": f"{dataset}_1871",
    "d_files": 1,
    "batch_size": 32,
    "learning_rate": 2e-3,
        },
}

model = models[n]

bs = model['batch_size']
lr = model['learning_rate']
n_iters = int(model['max_tokens']) // (2048*bs*ga)
eval_interval = n_iters//(10*max_overtrain//4)
warmup = n_iters // (20*max_overtrain)

for i in range(0, max_overtrain*20+1, 8):
    if i == 0:
        name = f'backbone_{version}'
        init_from = 'scratch_0'
        max_iters = n_iters
        evint = eval_interval
        decay_lr = False
    else:
        name = f'm{str(i)}_{version}'
        init_from = 'resume_0'
        max_iters = int(eval_interval * i // 8 / 0.8)
        evint = max_iters
        decay_lr = True

    runid = f"lrtest_{name}_fw_s{n}M_v{model['vocab_size']}_d{model['n_embd']}_l{model['n_layer']}_lin_bs{bs*ga}_wm5p_lr{lr}_b2{b2}"
    config = f"""
out_dir = 'out-{runid}'
eval_interval = {evint} # keep frequent because we'll overfit
eval_iters = 200
log_interval = 10 # don't print too too often

always_save_checkpoint = True

wandb_log = True # override via command line if you like
wandb_project = 'scaling_laws'
wandb_run_name = '{runid}'
log_activations = False
init_from = '{init_from}'

dataset = "{model['dataset']}"
data_files = {model['d_files']}
gradient_accumulation_steps = {ga}
batch_size = {bs}
block_size = 2048

vocab_size = {model['vocab_size']}

# baby GPT model :)
n_layer = {model['n_layer']}
n_head = {model['n_head']}
n_embd = {model['n_embd']}
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

    os.mkdir(f'out-{runid}')
    with open(f"train_{name}.py", "w") as f:
        f.write(config)
