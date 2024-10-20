import json, os
from transformers import DataCollatorForLanguageModeling, AutoTokenizer
from torch.utils.data import DataLoader

from datasets import load_from_disk
def log(text, dir = 'log.txt'):
    with open(dir, 'a+') as f:
        f.write(text)

def inspect(name):
    dataset = load_from_disk(name)['train']
    tokenizer = AutoTokenizer.from_pretrained('../models/pythia-160m')
    tokenizer.pad_token = tokenizer.eos_token
    collate_fn = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    data_loader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)
    for batch_idx, batch in enumerate(data_loader):
        print(f"Batch {batch_idx+1}")
        for key, value in batch.items():
            print(f"{key}: {value}, shape: {value.shape}")
        
        # 只检查前5个批次，防止打印过多内容
        if batch_idx == 4:
            break



if __name__ == "__main__":
    inspect('./outputs/dataset')