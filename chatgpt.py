import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Load pre-trained GPT-2 model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

# Define the generated sequence to evaluate
sequence = "This is a test sentence to calculate perplexity."
# Tokenize the sequence and add the special tokens
input_ids = tokenizer.encode(sequence, add_special_tokens=True, return_tensors='pt')

# Generate the predicted probabilities for the next token
with torch.no_grad():
    outputs = model(input_ids)
    logits = outputs.logits[0, :-1, :]  # exclude last token's logits
    target_ids = input_ids[0, 1:]      # exclude first token
    loss = torch.nn.functional.cross_entropy(logits, target_ids)
    
# Calculate perplexity
perplexity = torch.exp(loss)
print(f"Perplexity: {perplexity}")