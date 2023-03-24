from datasets import load_dataset
from transformers import AutoModel, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("a1noack/bart-large-gigaword")
model = AutoModel.from_pretrained("a1noack/bart-large-gigaword")
dataset = load_dataset("gigaword", split="test")
documents = dataset['document'][:50]
summaries = dataset['summary'][:50]

pass