import os
import pandas as pd
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments

# Load environment variables
load_dotenv()

# Load the Hugging Face token
token = os.getenv("HUGGINGFACE_TOKEN")

# Load training data
train_data = pd.read_csv('../data/processed_data.csv')  # Load your processed data

# Load tokenizer and model
model_name = "gemma-2b-it"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Tokenize input texts
train_encodings = tokenizer(train_data['processed_text'].tolist(), truncation=True, padding=True, return_tensors='pt')

# Prepare dataset
class ImpressionDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: val[idx] for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings['input_ids'])

train_dataset = ImpressionDataset(train_encodings)

# Set up training arguments
training_args = TrainingArguments(
    output_dir='./results',
    per_device_train_batch_size=2,
    num_train_epochs=2,
    logging_dir='./logs',
)

# Define Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset
)

# Fine-tune the model
trainer.train()
