from datasets import load_dataset
import random

# Load the CodeSearchNet Python dataset from Hugging Face
codesearchnet = load_dataset("Nan-Do/code-search-net-python")

# Shuffle the dataset and select a subset of 100,000 documents
python_docs = codesearchnet["train"]
shuffled_docs = python_docs.shuffle(seed=42)
subset_docs = shuffled_docs.select(range(100000))

# Save the subset for later use
subset_docs.save_to_disk("codesearchnet_subset")

print(len(subset_docs))  # Should output 100000