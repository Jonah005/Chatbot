import torch
import torch.nn as nn
import torch.optim as optim
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import json

model_save_path = "C:/Users/jonah/Allec2/models/transformer_model"
feedback_file = "C:/Users/jonah/Allec2/data/feedback_log.json"

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(768, 128)
        self.fc2 = nn.Linear(128, 1)  # One output for rating prediction

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x

# Load GPT-2 model
gpt2_model = GPT2LMHeadModel.from_pretrained(model_save_path)
gpt2_tokenizer = GPT2Tokenizer.from_pretrained(model_save_path)

# Load feedback data
def load_feedback():
    with open(feedback_file, 'r') as f:
        return json.load(f)

feedback_data = load_feedback()
neural_net = NeuralNetwork()

# Use AdamW optimizer for better fine-tuning based on feedback
optimizer = optim.AdamW(neural_net.parameters(), lr=0.001)

criterion = nn.MSELoss()

# Train neural network based on feedback responses
def train_neural():
    for entry in feedback_data["feedback"]:
        input_ids = gpt2_tokenizer(entry["user_input"], return_tensors="pt").input_ids
        response_ids = gpt2_tokenizer(entry["better_response"], return_tensors="pt").input_ids
        gpt2_output = gpt2_model(input_ids)[0]

        optimizer.zero_grad()

        # Predict and compute loss
        prediction = neural_net(gpt2_output)
        loss = criterion(prediction, response_ids.float())
        loss.backward()
        optimizer.step()

        print(f"Training loss: {loss.item()}")

# Ensure the function is called within the main scope
if __name__ == "__main__":
    train_neural()
