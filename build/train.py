import json as json
from utils.nltk_utils import tokenize, stem, bag_of_words
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from model import NeuralNet

with open('../resource/intents.json', 'r') as f:  # Open the intents.json file
    intents = json.load(f)

all_words = []
tags = []
xy = []

for intent in intents['intents']:  # For each intent in the intents
    tag = intent['tag']
    tags.append(tag)  # tags contains all tags in the dataset
    for pattern in intent['patterns']:  # For each pattern in the intent
        w = tokenize(pattern)  # Tokenize each sentence
        all_words.extend(w)  # all_words contains all words in the dataset
        xy.append((w, tag))  # xy contains a tuple with tokenized sentence and tag

ignore_words = ['?', '!', '.', ',']
all_words = [stem(w) for w in all_words if w not in ignore_words]  # Stem all words and remove ignore_words
all_words = sorted(set(all_words))  # Remove duplicates
tags = sorted(set(tags))  # Remove duplicates

x_train = []
y_train = []
for (pattern_sentence, tag) in xy:
    bag = bag_of_words(pattern_sentence, all_words)  # Bag of words for each sentence
    x_train.append(bag)  # x_train contains bag of words for each sentence

    label = tags.index(tag)  # Index of the tag
    y_train.append(label)  # y_train contains the index of the tag // Cross-Entropy Loss

x_train = np.array(x_train)
y_train = np.array(y_train)


class ChatDataset(Dataset):
    def __init__(self):
        self.n_samples = len(x_train)
        self.x_data = x_train
        self.y_data = y_train

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples


# Hyperparameters
batch_size = 8
hidden_size = 8
output_size = len(tags)
input_size = len(x_train[0])
learning_rate = 0.001
num_epochs = 2500

dataset = ChatDataset()
train_loader = DataLoader(dataset=dataset, batch_size=8, shuffle=True, num_workers=0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = NeuralNet(input_size=len(x_train[0]), hidden_size=hidden_size, number_classes=len(tags))

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the build
for epoch in range(num_epochs):
    for (words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(words)
        loss = criterion(outputs, labels)

        # Backward and optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f'[Training]  Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.10f}')


print(f'[Training] Final loss: {loss.item():.10f}')


"""
Save the build
"""
data = {
    "model_state": model.state_dict(),
    "input_size": input_size,
    "output_size": output_size,
    "hidden_size": hidden_size,
    "all_words": all_words,
    "tags": tags
}

FILE = "../data.pth"
torch.save(data, FILE)

print(f'Training complete. File saved to {FILE}')