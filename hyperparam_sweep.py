models = {
    5: {
    "n_layer": 6,
    "n_head": 4,
    "n_embd": 224,
    "vocab_size": 2000,
    "max_tokens": 5e6*20*4 // (1024*256) * (1024*256),
    "dataset": "fineweb2",
    "d_files": 2,
        },
    16: {
    "n_layer": 12,
    "n_head": 5,
    "n_embd": 320,
    "vocab_size": 5000,
    "max_tokens": 16e6*20*4 // (1024*256) * (1024*256),
    "dataset": "fineweb5",
    "d_files": 2,
        },
    47: {
    "n_layer": 18,
    "n_head": 7,
    "n_embd": 448,
    "vocab_size": 9000,
    "max_tokens": 47e6*20*4 // (1024*256) * (1024*256),
    "dataset": "fineweb9",
    "d_files": 5,
        }
}

batch_sizes = [16, 32, 64, 128]
beta2 = [0.95]
lrs = [6e-4, 1.2e-3, 2.4e-3, 4.8e-3, 1e-2]

ga = 1

for n, params in [(k, models[k]) for k in [47,16,5]]:
    i = 0
    for bs in batch_sizes:
        if n == 47 and bs > 64:
            ga = bs // 64
            bs = 64
        for b2 in beta2:
            for lr in lrs:
                n_iters = int(params['max_tokens']) // (1024*bs*ga)
                eval_interval = n_iters
                warmup = n_iters // 80
                config = f"""
out_dir = 'out-s{n}M_v{params['vocab_size']}_d{params['n_embd']}_l{params['n_layer']}_lin_bs{bs*ga}_wm5p_lr{lr}_b2{b2}'
eval_interval = {eval_interval} # keep frequent because we'll overfit
eval_iters = 200
log_interval = 10 # don't print too too often

always_save_checkpoint = True

wandb_log = True # override via command line if you like
wandb_project = 'scaling_laws'
wandb_run_name = 'sweep2_s{n}M_v{params['vocab_size']}_d{params['n_embd']}_l{params['n_layer']}_lin_bs{bs*ga}_wm5p_lr{lr}_b2{b2}'

dataset = "{params['dataset']}"
data_files = {params['d_files']}
gradient_accumulation_steps = {ga}
batch_size = {bs}
block_size = 1024

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
