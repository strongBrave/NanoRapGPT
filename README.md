# NanoRapGPT (finetuned from GPT-2)

![nanoGPT](assets/NanoRapGPT.png)

This repository is a fine-tuned version of the original minGPT project. I have used a custom dataset of rap lyrics that I crawled to fine-tune a GPT-2 model, transforming it into a rap lyrics generator. The code is simple and clean: train.py contains a ~300-line training loop, and model.py is a ~300-line definition of the GPT model, which can optionally load pre-trained GPT-2 weights from OpenAI. The model is still in development, but it already generates creative and unique rap lyrics. Currently, the training process is running on a single 8XA100 40GB node and should take about 4 days to train. Still under active development!

![repro124m](assets/gpt2_124M_loss.png)

Because the code is so simple, it is very easy to hack to your needs, train new models from scratch, or finetune pretrained checkpoints (e.g. biggest one currently available as a starting point would be the GPT-2 1.3B model from OpenAI).

## install

```
pip install torch numpy transformers datasets tiktoken wandb tqdm lyricsgenius
```

Dependencies:

- [pytorch](https://pytorch.org) <3
- [numpy](https://numpy.org/install/) <3
-  `transformers` for huggingface transformers <3 (to load GPT-2 checkpoints)
-  `datasets` for huggingface datasets <3 (if you want to download + preprocess OpenWebText)
-  `tiktoken` for OpenAI's fast BPE code <3
-  `wandb` for optional logging <3
-  `tqdm` for progress bars <3

## lyrics data generation

To generate lyrics data, follow these steps:

### Step 1: Crawl Lyrics Data
Run the `crawl.py` script to fetch lyrics data from the Genius API. You can specify the target artists either through a file or directly via command-line arguments.

```sh
python data/lyrics/data_generator/crawl.py --user_token=<your_genius_api_token> \
                                           --data_dir=<output_directory> \
                                           --max_songs=<max_songs_per_artist> \
                                           --artist_file=<path_to_artist_file> \
                                           --artists <artist1> <artist2> ...
```

**Key Parameters:**
- `--user_token`: Your Genius API user token (required).
- `--data_dir`: Directory to save the crawled data (default: `json`).
- `--max_songs`: Maximum number of songs to fetch per artist (default: `40`).
- `--artist_file`: Path to a file containing artist names (one per line).
- `--artists`: List of artist names provided directly via the command line.

### Step 2: Preprocess Lyrics Data
Run the `preprocess.py` script to clean and process the crawled lyrics data. This will save the processed data into a single JSON file.

```sh
python data/lyrics/data_generator/preprocess.py --data_dir=<input_directory> \
                                                --save_dir=<output_directory>
```

**Key Parameters:**
- `--data_dir`: Directory containing the crawled data (default: `json`).
- `--save_dir`: Directory to save the processed `data.json` file.

The final `data.json` file will be saved in the specified `save_dir` and can be used for training or fine-tuning the model.

### Step 3: Prepare Binary Files
Run the `prepare.py` script to convert the processed `data.json` file into binary files (`train.bin` and `val.bin`) for training.

```sh
python data/lyrics/prepare.py
```

This script will:
1. Load the `data.json` file.
2. Split the dataset into training and validation sets.
3. Tokenize the lyrics using the GPT-2 tokenizer.
4. Save the tokenized data into binary files (`train.bin` and `val.bin`).

The binary files will be saved in the same directory as the `prepare.py` script and can be used for training/finetuning the model.

## finetuning

Finetuning is no different than training, we just make sure to initialize from a pretrained model and train with a smaller learning rate. For an example of how to finetune a GPT on new text go to `data/shakespeare` and run `prepare.py` to download the tiny shakespeare dataset and render it into a `train.bin` and `val.bin`, using the OpenAI BPE tokenizer from GPT-2. Unlike OpenWebText this will run in seconds. Finetuning can take very little time, e.g. on a single GPU just a few minutes. Run an example finetuning like:

```sh
python train.py config/finetune_lyrics.py
```

This will load the config parameter overrides in `config/finetune_lyrics.py` (I didn't tune them much though). Basically, we initialize from a GPT2 checkpoint with `init_from` and train as normal, except shorter and with a small learning rate. If you're running out of memory try decreasing the model size (they are `{'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}`) or possibly decreasing the `block_size` (context length). The best checkpoint (lowest validation loss) will be in the `out_dir` directory, e.g. in `out-lyrics` by default, per the config file. You can then run the code in `sample.py --out_dir=out-lyrics`:

```
THEODORE:
Thou shalt sell me to the highest bidder: if I die,
I sell thee to the first; if I go mad,
I sell thee to the second; if I
lie, I sell thee to the third; if I slay,
I sell thee to the fourth: so buy or sell,
I tell thee again, thou shalt not sell my
possession.

JULIET:
And if thou steal, thou shalt not sell thyself.

THEODORE:
I do not steal; I sell the stolen goods.

THEODORE:
Thou know'st not what thou sell'st; thou, a woman,
Thou art ever a victim, a thing of no worth:
Thou hast no right, no right, but to be sold.
```

Whoa there, GPT, entering some dark place over there. I didn't really tune the hyperparameters in the config too much, feel free to try!

## sampling / inference

Use the script `sample.py` to sample either from pre-trained GPT-2 models released by OpenAI, or from a model you trained yourself. For example, here is a way to sample from the largest available `gpt2-xl` model:

```sh
python sample.py \
    --init_from=gpt2-xl \
    --start="What is the answer to life, the universe, and everything?" \
    --num_samples=5 --max_new_tokens=100
```

If you'd like to sample from a model you trained, use the `--out_dir` to point the code appropriately. You can also prompt the model with some text from a file, e.g. ```python sample.py --start=FILE:prompt.txt```.

## efficiency notes

For simple model benchmarking and profiling, `bench.py` might be useful. It's identical to what happens in the meat of the training loop of `train.py`, but omits much of the other complexities.

Note that the code by default uses [PyTorch 2.0](https://pytorch.org/get-started/pytorch-2.0/). At the time of writing (Dec 29, 2022) this makes `torch.compile()` available in the nightly release. The improvement from the one line of code is noticeable, e.g. cutting down iteration time from ~250ms / iter to 135ms / iter. Nice work PyTorch team!

## todos

- Investigate and add FSDP instead of DDP
- Eval zero-shot perplexities on standard evals (e.g. LAMBADA? HELM? etc.)
- Finetune the finetuning script, I think the hyperparams are not great
- Schedule for linear batch size increase during training
- Incorporate other embeddings (rotary, alibi)
- Separate out the optim buffers from model params in checkpoints I think
- Additional logging around network health (e.g. gradient clip events, magnitudes)
- Few more investigations around better init etc.

## troubleshooting

Note that by default this repo uses PyTorch 2.0 (i.e. `torch.compile`). This is fairly new and experimental, and not yet available on all platforms (e.g. Windows). If you're running into related error messages try to disable this by adding `--compile=False` flag. This will slow down the code but at least it will run.

For some context on this repository, GPT, and language modeling it might be helpful to watch my [Zero To Hero series](https://karpathy.ai/zero-to-hero.html). Specifically, the [GPT video](https://www.youtube.com/watch?v=kCc8FmEb1nY) is popular if you have some prior language modeling context.

For more questions/discussions feel free to stop by **#nanoGPT** on Discord:

[![](https://dcbadge.vercel.app/api/server/3zy8kqD9Cp?compact=true&style=flat)](https://discord.gg/3zy8kqD9Cp)

## acknowledgements

All nanoGPT experiments are powered by GPUs on [Lambda labs](https://lambdalabs.com), my favorite Cloud GPU provider. Thank you Lambda labs for sponsoring nanoGPT!
