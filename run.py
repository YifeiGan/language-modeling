
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_dataset
import numpy as np
import sys

output_path = sys.argv[1]

# Load Penn Treebank dataset
ptb = load_dataset('ptb-text-only/ptb_text_only')

# Tokenize sentences and build vocabulary
class Vocabulary:
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []
    
    def build_vocab(self, sentences):
        for sentence in sentences:
            for word in sentence.split():
                if word not in self.word2idx:
                    self.idx2word.append(word)
                    self.word2idx[word] = len(self.word2idx)

    def encode(self, sentence):
        return [self.word2idx.get(word, self.word2idx['<unk>']) for word in sentence.split()]

    def decode(self, indices):
        return [self.idx2word[idx] for idx in indices]

# Prepare dataset
def prepare_data(ptb_split, vocab):
    sentences = [entry['sentence'] for entry in ptb_split]
    vocab.build_vocab(sentences)
    data = []
    for sentence in sentences:
        tokens = vocab.encode(sentence)
        data.extend(tokens)
    return data

# Initialize vocabulary and prepare data
vocab = Vocabulary()
vocab.build_vocab(["<unk>"])  # Add unknown token to vocabulary

train_data = prepare_data(ptb['train'], vocab)
validation_data = prepare_data(ptb['validation'], vocab)
test_data = prepare_data(ptb['test'], vocab)

# Create data loader
def get_batches(data, batch_size, seq_length):
    n = len(data) // (batch_size * seq_length)
    data = data[:n * batch_size * seq_length]
    data = torch.tensor(data).view(batch_size, -1)
    for i in range(0, data.size(1), seq_length):
        inputs = data[:, i:i + seq_length]
        targets = data[:, i + 1:i + seq_length + 1]
        if inputs.size(1) == seq_length and targets.size(1) == seq_length:
            yield inputs, targets

batch_size = 32
seq_length = 30

# Define the LSTM model
class LanguageModel(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super(LanguageModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, x, hidden):
        x = self.embedding(x)
        output, hidden = self.lstm(x, hidden)
        logits = self.fc(output)
        return logits, hidden

# Model initialization
embed_size = 100
hidden_size = 256
vocab_size = len(vocab.idx2word)
model = LanguageModel(vocab_size, embed_size, hidden_size)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Initialize perplexities list
perplexities = []

for epoch in range(epochs):
    hidden = None
    epoch_loss = 0
    for inputs, targets in get_batches(train_data, batch_size, seq_length):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        
        # Forward pass
        logits, hidden = model(inputs, hidden)
        hidden = tuple([h.detach() for h in hidden])  # Detach hidden states to prevent backpropagation through time
        
        # Compute loss
        loss = criterion(logits.view(-1, vocab_size), targets.view(-1))
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        
    print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / len(train_data):.4f}")

import csv

# Function to compute perplexity for a single sentence
def compute_sentence_perplexity(sentence):
    model.eval()
    tokens = vocab.encode(sentence)
    inputs = torch.tensor(tokens).unsqueeze(0).to(device)
    targets = torch.tensor(tokens[1:] + [tokens[0]]).unsqueeze(0).to(device)
    hidden = None
    with torch.no_grad():
        logits, hidden = model(inputs, hidden)
        loss = criterion(logits.view(-1, vocab_size), targets.view(-1))
    perplexity = np.exp(loss.item())
    return perplexity

# Compute perplexities for each sentence in the test set
test_sentences = [entry['sentence'] for entry in ptb['test']]
perplexities = [compute_sentence_perplexity(sentence) for sentence in test_sentences]
id = 0

# Save the results into a CSV file
with open(output_path, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['ID', 'ppl'])
    for sentence, perplexity in zip(test_sentences, perplexities):
        writer.writerow([id, perplexity])
        id += 1
print("Perplexities saved to", output_path)
