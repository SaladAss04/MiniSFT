from transformers import  TrainingArguments, Trainer, DataCollatorForLanguageModeling

def train_instruction_following(model, tokenizer, dataset):
    training_args = TrainingArguments(
        output_dir="./outputs",
        eval_strategy="epoch",
        per_device_train_batch_size = 2,
        per_device_eval_batch_size = 4,
        learning_rate=2e-5,
        weight_decay=0.01,
        push_to_hub=True
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
