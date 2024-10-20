from datasets import load_dataset
import torch
from .test.misc import log

def preprocess_dataset(data, tokenizer, max_length):
    """
    Takes in shareGPT-styled data:
    {id: ..., conversations: [{from: human, value: '...'}, {from: gpt, value: '...'}]},
    Outputs labeled Mistral data:
    {text: <s>[INST]...[\INST]...[INST]...[\INST]...<\s>}
    """
    def fit_template(example):
        template = "[INST]{ins}[\\INST]{ans}<\\s>"
        ret = "<s>"
        example["answers"] = []
        for i in range(0, len(example['conversations'])-1, 2):
            if not example["conversations"][i]["from"] == "human" and example["conversations"][i+1]["from"] == "gpt":
                i += 1
            instruction, answer = example["conversations"][i], example["conversations"][i+1]
            mistral = template.format(ins = instruction["value"], ans = answer["value"])
            ret += mistral
            example["answers"].append(answer["value"])
        example['text'] = ret
        return example

    template_dataset = data.map(fit_template, remove_columns=['conversations', 'source'], num_proc=1) 
    
    def preprocess(example):
        inputs = tokenizer(example["text"], truncation=True, padding = 'max_length', max_length = max_length, return_tensors="pt", return_offsets_mapping = True)
        labels = torch.ones_like(inputs["input_ids"]) * -100
        #log("input ids:" + inputs["input_ids"][0] + "\n\n\n")
        log_template = """
        for this entry, there's {n} answers
        ***********************************
        labelling text:

        {text}
        -----------------------------------
        labelling answer:

        {answer}
        -----------------------------------
        (char_start, char_end), (token_start, token_end):
        
        ({cs},{ce}), ({ts},{te})
        ----------------------------------- 
        labelled:
        ({label})
        ***********************************
        """ 
        for answer in example["answers"]:
            char_start = example["text"].find(answer)
            char_end = char_start + len(answer) - 1
            
            token_start = 0
            token_end = 0
            for i, x in enumerate(inputs['offset_mapping'][0]):
                start, end = x[0], x[1]
                if start <= char_start < end:
                    token_start= i
                if start < char_end <= end:
                    token_end= i
                    labels[0][token_start:token_end + 1] = inputs["input_ids"][0][token_start:token_end + 1]
                    break
            if token_end == 0 and token_start != 0:
                labels[0][token_start:] = inputs["input_ids"][0][token_start:]
        ''' 
            log(log_template.format(n=len(example["answers"]), text = example["id"], answer = answer, cs = char_start, ce = char_end,
                                    ts = token_start, te = token_end, label = str(labels[0].tolist())))
        '''
        #THIS IS KEY
        inputs["labels"] = labels
        inputs["input_ids"] = inputs["input_ids"][0]
        inputs["labels"] = inputs["labels"][0]
        inputs["attention_mask"] = inputs["attention_mask"][0]
        inputs["offset_mapping"] = inputs["offset_mapping"][0]
        #print(inputs["input_ids"].shape, inputs["labels"].shape, inputs["attention_mask"].shape)
        return inputs
    
    tokenized_dataset = template_dataset.map(preprocess, remove_columns=['id', 'answers', 'text'], batched = False, num_proc=8)
    return tokenized_dataset

def prepare_dataset(name, tokenizer, max_context = 2048):
    """
    The master function for return train&test dataset.
    """
    raw_dataset = load_dataset(name)['train']
    tokenized_dataset = preprocess_dataset(raw_dataset, tokenizer, max_context)
    tokenized_dataset = tokenized_dataset.train_test_split(test_size=0.1)
    return tokenized_dataset