from transformers import  TrainingArguments, Trainer, DataCollatorForLanguageModeling
import nltk
from nltk.translate.bleu_score import sentence_bleu

def train_instruction_following(model, tokenizer, dataset):
    def compute_metrics(pred):
        references = pred.label_ids
        generated_texts = pred.predictions
        
        bleu_scores = []
        for reference, generated_text in zip(references, generated_texts):
            reference_text = train_dataset[reference]['text']
            bleu_score = sentence_bleu([reference_text], generated_text)
            bleu_scores.append(bleu_score)

        return {
            'bleu': sum(bleu_scores) / len(bleu_scores)
        }
    training_args = TrainingArguments(
        output_dir="./outputs/model/zero2",
        eval_strategy="epoch",
        per_device_train_batch_size = 8,
        per_device_eval_batch_size = 2,
        learning_rate=8e-6,
        weight_decay=1e-5,
        push_to_hub=False,
        num_train_epochs=8,
        max_grad_norm = 2.0,
        logging_dir="./outputs/model/new/logs",
        logging_steps=20,
        save_steps=500,
        eval_steps=500,
        deepspeed="./src/configs/ds_s2.json",  # 启用 DeepSpeed 并指定配置文件
        report_to="tensorboard",
        fp16=True
    )
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    trainer.train()
