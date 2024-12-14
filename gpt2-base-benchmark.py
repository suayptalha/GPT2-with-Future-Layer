from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
import torch.nn as nn
from datasets import load_dataset
from nltk.translate.bleu_score import sentence_bleu
import math

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

def calculate_perplexity(model, input_ids, labels):
    with torch.no_grad():
        outputs = model(input_ids=input_ids, labels=labels)
        log_likelihood = outputs[0]
        perplexity = torch.exp(log_likelihood.mean())
    return perplexity.item()

def calculate_bleu_score(reference, generated):
    reference = reference.split()
    generated = generated.split()
    return sentence_bleu([reference], generated)

dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split='test')

perplexity_scores = []
bleu_scores = []

for i in range(100):
    sample_text = dataset[i]['text']

    if len(sample_text) < 25:
        print(f"Skipping sample {i+1} as its length is less than 25 characters.")
        continue

    input_ids = tokenizer(sample_text, return_tensors="pt").input_ids
    
    if input_ids.shape[1] == 0:
        print(f"Skipping sample {i+1} due to empty input_ids")
        continue
    
    if input_ids.shape[1] > 128:
        print(f"Skipping sample {i+1} as its length exceeds 128 tokens.")
        continue

    labels = input_ids.clone()

    perplexity = calculate_perplexity(model, input_ids, labels)
    perplexity_scores.append(perplexity)

    generated_output = model.generate(input_ids=input_ids, max_length=128, num_beams=5, no_repeat_ngram_size=2, top_k=50, top_p=0.95)

    generated_text = tokenizer.decode(generated_output[0], skip_special_tokens=True)

    reference_text = sample_text 
    bleu_score_value = calculate_bleu_score(reference_text, generated_text)
    bleu_scores.append(bleu_score_value)

    print(f"Sample {i + 1}:")
    print(f"Generated Text: {generated_text}")
    print(f"Perplexity: {perplexity}")
    print(f"BLEU Score: {bleu_score_value}")
    print("-" * 50)

average_perplexity = sum(perplexity_scores) / len(perplexity_scores)
average_bleu_score = sum(bleu_scores) / len(bleu_scores)

print(f"Average Perplexity: {average_perplexity}")
print(f"Average BLEU Score: {average_bleu_score}")
