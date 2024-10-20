from src.dataset import prepare_dataset
from src.train_script import train_instruction_following
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from datasets import load_dataset, load_from_disk

def main(regenerate = False):
    model = AutoModelForCausalLM.from_pretrained('../models/pythia-160m')
    toke = AutoTokenizer.from_pretrained('../models/pythia-160m')
    toke.pad_token = toke.eos_token
    if regenerate:
        '''
        raw_dataset = load_dataset('hkust-nlp/deita-6k-v0')['train']
        tokenized_dataset = prepare_dataset(raw_dataset, toke)
        '''
        tokenized_dataset = prepare_dataset('hkust-nlp/deita-6k-v0', toke)
        tokenized_dataset.save_to_disk('./outputs/dataset')
    else:
        tokenized_dataset = load_from_disk('./outputs/dataset')
    train_instruction_following(model, toke, tokenized_dataset)

def inference(model_name_or_path):
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    model.eval()

    def chat(model, tokenizer, input_text, max_length=2048):
        messages = [{"role":"user", "content":input_text}]
        pipe = pipeline("text-generation", model = model, tokenizer = tokenizer, device=0)
        tokenizer.chat_template = "<s><INST>{input_text}<\\INST>"
        generated = pipe(
            messages,
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id,
            max_length=2048,
        )
        messages = generated[0]["generated_text"]
        
        generated_text = messages[-1]["content"]
        
        return generated_text

    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Chatbot: 再见！")
            break
        
        response = chat(model, tokenizer, user_input)
        print(f"Chatbot: {response}")
             
if __name__ == "__main__":
    main(True) 
    #inference("./outputs/model/zero2/checkpoint-1013")