import json
import os
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AdamW
import logging

# File paths
feedback_file = "C:/Users/jonah/Allec2/data/feedback_log.json"
intents_file = "C:/Users/jonah/Allec2/data/intents.json"
model_save_path = "C:/Users/jonah/Allec2/models/transformer_model"

# Logging setup
logging.basicConfig(filename="feedback_processing.log", level=logging.INFO,
                    format="%(asctime)s:%(levelname)s:%(message)s")

# Load GPT-2 model and tokenizer
gpt2_model = GPT2LMHeadModel.from_pretrained(model_save_path)
gpt2_tokenizer = GPT2Tokenizer.from_pretrained(model_save_path)
gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token
optimizer = AdamW(gpt2_model.parameters(), lr=5e-5)


# Feedback processing to improve model based on feedback
def process_feedback(feedback_data):
    user_input = feedback_data.get("user_input")
    better_response = feedback_data.get("better_response")
    correct_tags = feedback_data.get("correct_tags", [])

    # Update feedback log
    if not os.path.exists(feedback_file):
        feedback_log = {"feedback": []}
    else:
        with open(feedback_file, "r") as f:
            feedback_log = json.load(f)

    feedback_log["feedback"].append(feedback_data)

    with open(feedback_file, "w") as f:
        json.dump(feedback_log, f, indent=4)

    logging.info(f"Processed feedback for user input: '{user_input}'")

    # Fine-tune the model using feedback
    fine_tune_model(user_input, better_response)

    # Update correct tags to intents.json if provided
    if correct_tags:
        update_intents_with_tags(user_input, correct_tags)


# Fine-tune the GPT-2 model using user feedback
def fine_tune_model(user_input, better_response):
    gpt2_model.train()
    input_ids = gpt2_tokenizer(user_input, return_tensors="pt").input_ids
    labels = gpt2_tokenizer(better_response, return_tensors="pt").input_ids

    # Ensure input and labels are the same length
    max_length = max(input_ids.shape[1], labels.shape[1])
    input_ids = torch.nn.functional.pad(input_ids, (0, max_length - input_ids.shape[1]),
                                        value=gpt2_tokenizer.pad_token_id)
    labels = torch.nn.functional.pad(labels, (0, max_length - labels.shape[1]), value=gpt2_tokenizer.pad_token_id)

    # Run training steps
    for step in range(5):  # Smaller number of steps to make updates faster
        optimizer.zero_grad()
        outputs = gpt2_model(input_ids=input_ids, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

    # Save updated model
    gpt2_model.save_pretrained(model_save_path)
    gpt2_tokenizer.save_pretrained(model_save_path)

    logging.info(f"Model fine-tuned based on feedback: '{user_input}' -> '{better_response}'")
    print(f"Model fine-tuned based on feedback: {user_input} -> {better_response}")


# Update intents.json file with correct tags from feedback
def update_intents_with_tags(user_input, correct_tags):
    if os.path.exists(intents_file):
        with open(intents_file, "r") as f:
            intents_data = json.load(f)

        # Check if the user input is already present in the intents
        for intent in intents_data['intents']:
            if user_input in intent['patterns']:
                # Add correct tags if missing
                intent['tag'] = list(set(intent['tag'] + correct_tags))  # Prevent duplicate tags

        # Save updated intents
        with open(intents_file, "w") as f:
            json.dump(intents_data, f, indent=4)

        logging.info(f"Updated intents.json with correct tags: {correct_tags} for input: {user_input}")
        print(f"Updated intents.json with tags: {correct_tags} for input: {user_input}")


if __name__ == "__main__":
    # Example feedback processing
    example_feedback = {
        "user_input": "How do I track my order?",
        "better_response": "Use the 'Order Tracking' section in the app to check your order status.",
        "correct_tags": ["order_tracking"]
    }
    process_feedback(example_feedback)
