import torch
import torch.nn as nn
from transformers import GPT2Tokenizer, GPT2LMHeadModel
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

    def generate_with_future(self, input_ids, max_length=128):
        generated_ids = input_ids.clone()
        for _ in range(max_length - input_ids.shape[1]):
            gpt2_outputs = self.gpt2_model(input_ids=generated_ids, output_hidden_states=True)
            current_features = gpt2_outputs.hidden_states[-1]
            future_features, _ = self.future_predictor(current_features)
            combined_features = current_features + 0.5 * future_features
            logits = self.gpt2_model.lm_head(combined_features[:, -1, :])
            next_token = torch.argmax(logits, dim=-1).unsqueeze(0)
            generated_ids = torch.cat([generated_ids, next_token], dim=1)
        return generated_ids

embed_dim = 768
num_heads = 12
num_layers = 4
gpt2_with_future_model = GPT2WithFuture(model, embed_dim=embed_dim, num_heads=num_heads, num_layers=num_layers)

def calculate_perplexity(model, input_ids, labels):
    with torch.no_grad():
        _, total_loss, _ = model(input_ids=input_ids, labels=labels)
        perplexity = torch.exp(total_loss.mean())
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
    if len(sample_text) < 25 or tokenizer(sample_text, return_tensors="pt").input_ids.shape[1] > 128:
        continue
    input_ids = tokenizer(sample_text, return_tensors="pt").input_ids
    labels = input_ids.clone()
    perplexity = calculate_perplexity(gpt2_with_future_model, input_ids, labels)
    perplexity_scores.append(perplexity)
    generated_output = gpt2_with_future_model.generate_with_future(input_ids, max_length=128)
    generated_text = tokenizer.decode(generated_output[0], skip_special_tokens=True)
    bleu_scores.append(calculate_bleu_score(sample_text, generated_text))

average_perplexity = sum(perplexity_scores) / len(perplexity_scores)
average_bleu_score = sum(bleu_scores) / len(bleu_scores)
print(f"Average Perplexity: {average_perplexity}")
print(f"Average BLEU Score: {average_bleu_score}")
