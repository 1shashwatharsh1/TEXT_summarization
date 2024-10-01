from transformers import AutoTokenizer

model_name = "gemma-2b-it"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token='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')

from dotenv import load_dotenv
load_dotenv()
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
from evaluate import load

# Load model name and token from environment variables
model_name = "gemma-2b-it"
token = os.getenv("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")  # Ensure your token is stored in the .env file

# Load evaluation data
eval_data = pd.read_csv('../data/eval_data.csv')

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Generate impressions for evaluation samples
def generate_impressions(samples):
    inputs = tokenizer(samples, return_tensors='pt', padding=True, truncation=True)
    outputs = model.generate(**inputs)
    return [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]

# Generate impressions
eval_samples = eval_data['Impression'].tolist()  # Assuming 'Impression' is the column for input text
generated_impressions = generate_impressions(eval_samples)

# Compute evaluation metrics
inputs = tokenizer(eval_samples, return_tensors='pt', padding=True, truncation=True)
loss = model(input_ids=inputs['input_ids'], labels=inputs['input_ids']).loss
perplexity = torch.exp(loss)

# ROUGE score calculation using the evaluate library
rouge = load("rouge")
results = rouge.compute(predictions=generated_impressions, references=eval_data['Impression'].tolist())  # Adjust if needed

print("Perplexity:", perplexity.item())
print("ROUGE Scores:", results)
