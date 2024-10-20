from src.dataset import prepare_dataset
from src.train_script import train_instruction_following
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from datasets import load_dataset, load_from_disk
import argparse
import os

def main(force_regenerate = False):
    '''
    model = AutoModelForCausalLM.from_pretrained('EleutherAI/pythia-160m')
    toke = AutoTokenizer.from_pretrained('EleutherAI/pythia-160m')
    '''
    toke = AutoTokenizer.from_pretrained('../models/pythia-160m')
    toke.pad_token = toke.eos_token
    if force_regenerate or not os.path.exists('./outputs/dataset'):
        '''
        raw_dataset = load_dataset('hkust-nlp/deita-6k-v0')['train']
        tokenized_dataset = prepare_dataset(raw_dataset, toke)
        '''
        tokenized_dataset = prepare_dataset('hkust-nlp/deita-6k-v0', toke)
        tokenized_dataset.save_to_disk('./outputs/dataset')
    else:
        tokenized_dataset = load_from_disk('./outputs/dataset')
        
    train_instruction_following('../models/pythia-160m', toke, tokenized_dataset)

def inference(model_name_or_path):
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model.eval()

    def chat_with_pipeline(model, tokenizer, input_text, max_length=2048):
        messages = [{"role":"user", "content":input_text}]
        pipe = pipeline("text-generation", model = model, tokenizer = tokenizer, device=0)
        tokenizer.chat_template = "<s><INST>{input_text}<\\INST>"
        generated = pipe(
            messages,
            eos_token_id=tokenizer.eos_token_id,
            max_length=50,
        )
        messages = generated[0]["generated_text"]
        
        generated_text = messages[-1]["content"]
        
        return generated_text
    
    def chat(model, tokenizer, input_text, max_length=2048):
        inputs = tokenizer(input_text, return_tensors="pt", padding=True)
        # 对输入数据进行推理
        outputs = model.generate(inputs['input_ids'], attention_mask=inputs['attention_mask'], max_length=100, num_beams=5, early_stopping=True)

        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        return generated_text
    
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Chatbot: 再见！")
            break
        
        response = chat_with_pipeline(model, tokenizer, user_input)
        print(f"Chatbot: {response}")
             
if __name__ == "__main__":
    main() 
    #inference("./outputs/model/zero3/checkpoint-1352")