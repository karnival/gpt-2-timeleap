models = {
    1: {
    "n_layer": 4,
    "n_head": 2,
    "n_embd": 128,
    "vocab_size": 1168,
    "max_tokens": (1e6*20*4 // (2048*128)) * (2048*128),
    "dataset": "fineweb_1168",
    "d_files": 1,
        },
    3: {
    "n_layer": 6,
    "n_head": 3,
    "n_embd": 192,
    "vocab_size": 1539,
    "max_tokens": (3e6*20*4 // (2048*128)) * (2048*128),
    "dataset": "fineweb_1539",
    "d_files": 1,
        },
    10: {
    "n_layer": 12,
    "n_head": 4,
    "n_embd": 256,
    "vocab_size": 1871,
    "max_tokens": (10e6*20*4 // (2048*128)) * (2048*128),
    "dataset": "fineweb_1871",
    "d_files": 1,
        }
}

batch_sizes = [8, 12, 16, 24, 32, 48, 64, 96, 128]
beta2 = [0.95, 0.99]
lrs = [6e-4, 1.2e-3, 2.4e-3, 4.8e-3, 1e-2, 2e-2]

ga = 1

i = 0

for n, params in [(k, models[k]) for k in [1, 3, 10]]:
    for bs in batch_sizes:
        for b2 in beta2:
            for lr in lrs:
                n_iters = int(params['max_tokens']) // (2048*bs*ga)
                eval_interval = n_iters
                warmup = n_iters // 80
                config = f"""
out_dir = 'out-fw_s{n}M_v{params['vocab_size']}_d{params['n_embd']}_l{params['n_layer']}_lin_bs{bs*ga}_wm5p_lr{lr}_b2{b2}'
eval_interval = {eval_interval} # keep frequent because we'll overfit
eval_iters = 200
log_interval = 10 # don't print too too often

always_save_checkpoint = True

wandb_log = True # override via command line if you like
wandb_project = 'scaling_laws'
wandb_run_name = 'sweep_new_1_fw_s{n}M_v{params['vocab_size']}_d{params['n_embd']}_l{params['n_layer']}_lin_bs{bs*ga}_wm5p_lr{lr}_b2{b2}'

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

max_iters = {n_iters}
learning_rate = {lr}
lr_decay='linear'
lr_decay_iters = {n_iters} # make equal to max_iters usually
min_lr = {lr/10}

beta2 = {b2}

warmup_iters = {warmup}

weight_decay = 1e-4/learning_rate
z_loss = 1e-4
                """

                with open(f"config{i}.py", "w") as f:
                    f.write(config)
                i += 1





import subprocess
import queue
import threading
import time

MAX_GPUS = 8

def run_script(script, gpu_id):
    command = f"CUDA_VISIBLE_DEVICES={gpu_id} python3 {script}"
    process = subprocess.Popen(command, shell=True)
    process.wait()

def worker(task_queue, gpu_semaphore):
    while True:
        script = task_queue.get()
        if script is None:
            break
        with gpu_semaphore:
            gpu_id = gpu_semaphore.get_gpu()
            print(f"Running {script} on GPU {gpu_id}")
            run_script(script, gpu_id)
            print(f"Finished {script} on GPU {gpu_id}")
            gpu_semaphore.release_gpu(gpu_id)
        task_queue.task_done()

class GPUSemaphore:
    def __init__(self, count):
        self.semaphore = threading.Semaphore(count)
        self.available_gpus = queue.Queue()
        for i in range(count):
            self.available_gpus.put(i)

    def __enter__(self):
        self.semaphore.acquire()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.semaphore.release()
        
    def get_gpu(self):
        return self.available_gpus.get()

    def release_gpu(self, gpu_id):
        self.available_gpus.put(gpu_id)

def main(scripts):
    task_queue = queue.Queue()
    gpu_semaphore = GPUSemaphore(MAX_GPUS)

    # Start worker threads
    threads = []
    for _ in range(MAX_GPUS):
        t = threading.Thread(target=worker, args=(task_queue, gpu_semaphore))
        t.start()
        threads.append(t)

    # Add tasks to the queue
    for script in scripts:
        task_queue.put(script)

    # Add None to the queue to signal the workers to exit
    for _ in range(MAX_GPUS):
        task_queue.put(None)

    # Wait for all tasks to complete
    task_queue.join()

    # Wait for all threads to finish
    for t in threads:
        t.join()

if __name__ == "__main__":
    # Example usage
    scripts = [f"train.py config{i}.py" for i in range(4*5*3)]  # List of scripts to run
    main(scripts)
