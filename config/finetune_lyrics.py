import time

out_dir = 'out-lyrics'
eval_interval = 5
eval_iters = 100

dataset = 'lyrics'
init_from = 'gpt2-large' # this is the largest GPT-2 model

# only save checkpoints if the validation loss improves
always_save_checkpoint = False

# the number of examples per iter:
# 1 batch_size * 32 grad_accum * 1024 tokens = 32,768 tokens/iter
# shakespeare has 301,966 tokens, so 1 epoch ~= 9.2 iters
batch_size = 1
block_size = 914 # median number of tokens in the dataset
gradient_accumulation_steps = 32
max_iters = 5

# finetune at constant LR
learning_rate = 3e-5
decay_lr = False
dropout = 0.2

wandb_log = True # feel free to turn on
wandb_project = 'finetune_lyrics'
wandb_run_name = f'b{batch_size}_ga{gradient_accumulation_steps}'