---
title: SFT Implementation

---

## SFT Implementation
### A. How to reproduce
The project can be deployed as follows.
```
git clone https://github.com/SaladAss04/MiniSFT.git
cd MiniSFT
python3 -m sft
source sft/bin/activate
pip install -r requirements-pip.txt
```
After the installation of the environment, run the following script for dataset preprocessing and training using non-distributed vanilla trainer, deepspeed ZeRo-1, ZeRo-2, ZeRo-3, repsectively
```
./src/scripts/vanilla.sh
./src/scripts/ds1.sh
./src/scripts/ds2.sh
./src/scripts/ds3.sh
```
The resulting model will be saved to *./output/model/*.
To evaluate, finish trainig using the above scripts, then run
```
./src/scripts/evaluate.sh
```
BLEU-score and 3 example outputs of all checkpoints of the given model will be logged to *./outputs/evaluation/*
### B.Implementation
#### 1. Dataset Preprocessing
This is a instruction-fine-tuning task using Mistral-styled chat template, given training data in ShareGPT template. Therefore, after downloading and splitting the data into 'train' and 'test' (by 1:10), the first step is to fit the text into Mistral chat template, which is
> \<s\>\<INST\>instruction\<\INST\>response<\s>\<INST\>instruction\<\INST\>response<\s>...

This is simply implemented by filling a pair of instruction-response in the template once and concatenating, as in `src.dataset::fill_template(example)`.

Then perform tokenization and loss-masking (i.e. ignoring the loss of instructions). Taking advantage of one of huggingface's trainer's features, I achieved loss-masking by generating a 'label' tensor for each entry in the dataset, whose value is '-100' if the corresponding token is instruction and otherwise the corresponding token's id. This way, in trainer's loss calculation function, loss value of tokens labelled '-100' will be ignored. See this preprocessing function in `src.dataset::preprocess(example)`.
![Screenshot 2024-10-20 at 22.20.13](https://hackmd.io/_uploads/rJ-pb5MxJl.png)

Taking acount of the max context length, I simply allowed truncation during the tokenization step, because the given answers in the dataset are rather long and unaffected by truncation.
#### 2. Training Procedure
This implementation mainly utilized the huggingface trainer class, which included deepspeed implementation of distributed training. The training hyperparameters are listed as follows:


| Learning Rate | Epoch | Per Device Batch Size |Optimizer|Scheduler|Weight Decay
| -------- | -------- | -------- |-------- |-------- |-------- |
| 8e-6     | 8     | 8    |AdamW|Linear With Warm-up|1e-5

Given the small-parameter nature of our model, I chose samller learning rate and weight decay and performed a little more epoches. I wasn't able to experiment with many hyperparameters but these look good enough. I also allowed fp16 training for speed.
##### Deepspeed Stage
There are three stages for Deepspeed: ZeRO-1, ZeRO-2 and ZeRO-3. ZeRO-1 distributes only optimizer states, while ZeRO-2 shards both optimizer states and gradients, and ZeRO-3 shards optimizer states, gradients and parameters. 

I experimented the three stages with one epoch of initial trainibg; the loss curve is shown below. We can see that ZeRO-1 provides only tiny improvements to speed, while ZeRO-2 and ZeRO-3 shows similar improvements to regression speed. However, it can be inferred that ZeRO-2 will certainly provide better result, because for a small model like ours, distributing parameters between GPUs at the cost of frequent communication is not worth it.
### C.Results
#### 1. Training Loss Curve
#### 2. Generated Responses from Chat
#### 3. Bleu Scores and Example Outputs from Evaluation
