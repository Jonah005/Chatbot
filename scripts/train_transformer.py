import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments
import json
import os

# Paths to data and model
data_path = "C:/Users/jonah/Allec2/data/intents.json"
model_save_path = "C:/Users/jonah/Allec2/models/transformer_model"

# Function to load data from a JSON file
def load_data(data_file):
    with open(data_file, 'r') as f:
        return json.load(f)

# Custom dataset class to prepare data for GPT-2 model fine-tuning for SQL generation
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data['intents']  # Training data loaded from intents
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token  # Use EOS as PAD token if missing

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        intent = self.data[idx]
        prompt = intent['patterns'][0]  # User input
        response = intent['responses'][0]  # Corresponding SQL query

        # Tokenize the user input (prompt) and the SQL response (label)
        input_ids = self.tokenizer(prompt, return_tensors='pt', padding='max_length', max_length=150, truncation=True).input_ids[0]
        label_ids = self.tokenizer(response, return_tensors='pt', padding='max_length', max_length=150, truncation=True).input_ids[0]

        return {"input_ids": input_ids, "labels": label_ids}

# Load dataset from the JSON file
data = load_data(data_path)
dataset = CustomDataset(data)

# Load GPT-2 model for fine-tuning on SQL queries
model = GPT2LMHeadModel.from_pretrained('gpt2')

# Ensure the model save path directory exists
os.makedirs(model_save_path, exist_ok=True)

# Train GPT-2 to understand SQL structure
training_args = TrainingArguments(
    output_dir=model_save_path,
    per_device_train_batch_size=4,  # Increase batch size for faster training
    num_train_epochs=10,  # Increase the epochs to improve learning for complex SQL
    logging_dir='./logs',
    save_steps=500,  # Save model frequently to prevent data loss
    evaluation_strategy="steps",  # Evaluate periodically during training
    logging_steps=100,  # Log every 100 steps
    load_best_model_at_end=True,  # Load the best model at the end of training
)

# Trainer for fine-tuning the GPT-2 model
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset
)

# Train the model
trainer.train()  # Fine-tune GPT-2 on SQL queries

# Save the fine-tuned model and tokenizer
model.save_pretrained(model_save_path)
dataset.tokenizer.save_pretrained(model_save_path)

print(f"Model saved to {model_save_path}")
