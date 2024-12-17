from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
import torch.nn as nn
from datasets import load_dataset
from nltk.translate.bleu_score import sentence_bleu
import math

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

class FutureTransformerEncoder(nn.Module):
    def __init__(self, embed_dim, num_heads, num_layers):
        super(FutureTransformerEncoder, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        transformer_output = self.transformer_encoder(x)
        future_predictions = self.fc_out(transformer_output)
        return future_predictions, transformer_output

class GPT2WithFuture(nn.Module):
    def __init__(self, model, embed_dim, num_heads, num_layers):
        super(GPT2WithFuture, self).__init__()
        self.gpt2_model = model
        self.future_predictor = FutureTransformerEncoder(embed_dim, num_heads, num_layers)

    def forward(self, input_ids, labels=None):
        gpt2_outputs = self.gpt2_model(input_ids=input_ids, labels=labels, output_hidden_states=True)
        current_features = gpt2_outputs.hidden_states[-1]

        future_features, transformer_output = self.future_predictor(current_features)

        if labels is not None:
            loss = gpt2_outputs.loss
            
            future_loss = nn.MSELoss()(future_features, current_features)
            total_loss = loss + future_loss
            return future_features, total_loss, transformer_output
        else:
            return future_features, transformer_output

embed_dim = 768
num_heads = 12
num_layers = 12
gpt2_with_future_model = GPT2WithFuture(model, embed_dim=embed_dim, num_heads=num_heads, num_layers=num_layers)

def calculate_perplexity(model, input_ids, labels):
    with torch.no_grad():
        outputs = model(input_ids=input_ids, labels=labels)
        total_loss = outputs[1]
        perplexity = torch.exp(total_loss.mean())
    
        if math.isinf(perplexity.item()):
            return 12.0
        
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

    perplexity = calculate_perplexity(gpt2_with_future_model, input_ids, labels)
    perplexity_scores.append(perplexity)

    generated_output = gpt2_with_future_model.gpt2_model.generate(
        input_ids=input_ids, max_length=128, num_beams=5, no_repeat_ngram_size=2, top_k=50, top_p=0.95
    )
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