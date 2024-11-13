import json
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AdamW

# File paths
feedback_file = "C:/Users/jonah/Allec2/data/feedback_log.json"
model_save_path = "C:/Users/jonah/Allec2/models/transformer_model"

# Load GPT-2 model and tokenizer
gpt2_model = GPT2LMHeadModel.from_pretrained(model_save_path)
gpt2_tokenizer = GPT2Tokenizer.from_pretrained(model_save_path)
gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token

# AdamW optimizer for better learning performance
optimizer = AdamW(gpt2_model.parameters(), lr=5e-5)

# Reinforcement learning for SQL query fine-tuning based on feedback
def train_model_on_feedback():
    try:
        with open(feedback_file, "r") as f:
            feedback_data = json.load(f)

        for entry in feedback_data.get("feedback", []):
            if "user_input" in entry and "better_response" in entry:
                input_ids = gpt2_tokenizer(entry["user_input"], return_tensors="pt").input_ids
                labels = gpt2_tokenizer(entry["better_response"], return_tensors="pt").input_ids

                # Padding to ensure both input and labels have the same length
                max_length = max(input_ids.shape[1], labels.shape[1])
                input_ids = torch.nn.functional.pad(input_ids, (0, max_length - input_ids.shape[1]), value=gpt2_tokenizer.pad_token_id)
                labels = torch.nn.functional.pad(labels, (0, max_length - labels.shape[1]), value=gpt2_tokenizer.pad_token_id)

                optimizer.zero_grad()  # Reset gradients
                outputs = gpt2_model(input_ids=input_ids, labels=labels)
                loss = outputs.loss
                loss.backward()  # Compute gradients
                optimizer.step()  # Update model

                print(f"Processed feedback: '{entry['user_input']}' with feedback-driven response.")
                print(f"Training loss: {loss.item()}")

        # Save model after reinforcement learning
        gpt2_model.save_pretrained(model_save_path)
        gpt2_tokenizer.save_pretrained(model_save_path)

        print("Model fine-tuned using reinforcement learning based on feedback.")
    except Exception as e:
        print(f"Error during reinforcement learning: {e}")

if __name__ == "__main__":
    train_model_on_feedback()
