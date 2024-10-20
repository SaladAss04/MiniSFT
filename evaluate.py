import argparse
import os
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from sacrebleu import corpus_bleu
from tqdm import tqdm

def compute_bleu(predictions, references):
    # Compute BLEU-4 score using SacreBLEU
    bleu = corpus_bleu(predictions, [references], force=True)
    return bleu.score

def evaluate_model(model_path, dataset_path, output_path, code : str):
    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Load dataset
    dataset = load_dataset(dataset_path, split='test')

    predictions = []
    references = []

    # Prepare to store first 3 outputs
    first_3_outputs = []

    print("Evaluating model on test set...")
    for idx, sample in tqdm(enumerate(dataset), total=len(dataset)):
        input_text = sample['input_text']  # Assumed field name for input
        reference_text = sample['reference_text']  # Assumed field name for reference

        # Tokenize and generate prediction
        inputs = tokenizer(input_text, return_tensors="pt")
        outputs = model.generate(inputs['input_ids'], max_length=50)

        # Decode the model's output
        predicted_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Store predictions and references for BLEU calculation
        predictions.append(predicted_text)
        references.append(reference_text)

        # Save the first 3 outputs for inspection
        if idx < 3:
            first_3_outputs.append({
                'input': input_text,
                'predicted': predicted_text,
                'reference': reference_text
            })

    # Compute BLEU-4 score
    bleu_score = compute_bleu(predictions, references)
    print(f"BLEU-4 score: {bleu_score}")

    bleu_filename = 'bleu-score_' + code + "_" + model_path.split('/')[-1] + '.txt'
    # Save BLEU score to txt file
    with open(os.path.join(output_path, bleu_filename), 'w') as f:
        f.write(f"BLEU-4 score: {bleu_score}\n\n")

    output_filename ='output_' + code + "_" + model_path.split('/')[-1] + '.txt'
    # Save first 3 outputs to txt file
    with open(os.path.join(output_path, output_filename), 'w') as f:
        for idx, output in enumerate(first_3_outputs):
            f.write(f"Example {idx+1}:\n")
            f.write(f"Input: {output['input']}\n")
            f.write(f"Predicted: {output['predicted']}\n")
            f.write(f"Reference: {output['reference']}\n")
            f.write("\n")
    
    print("Evaluation complete. BLEU score and first 3 outputs saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Causal Language Model on BLEU-4 score")
    parser.add_argument('--model_path', type=str, required=True, help="Path to the pretrained model")
    parser.add_argument('--dataset_path', type=str, required=True, help="Path to the Huggingface dataset")
    parser.add_argument('--output_path', type=str, required=True, help="Directory to save the results")

    args = parser.parse_args()
    
    for dir in os.listdir(args.model_path):
        if dir.startswith('checkpoint'):
            evaluate_model(dir, args.dataset_path, args.output_path)