from src.dataset import prepare_dataset
from src.train_script import train_instruction_following
from transformers import AutoModelForCausalLM, AutoTokenizer
def main(regenerate = False):
    model = AutoModelForCausalLM.from_pretrained('../models/pythia-160m')
    toke = AutoTokenizer.from_pretrained('../models/pythia-160m')
    toke.pad_token = toke.eos_token
    if regenerate:
        tokenized_dataset = prepare_dataset('hkust-nlp/deita-6k-v0', toke)
        tokenized_dataset.save_to_disk('./outputs/dataset')
    else:
        tokenized_dataset = load_dataset('./outputs/dataset')
    train_instruction_following(model, toke, tokenized_dataset)

if __name__ == "__main__":
    main() 