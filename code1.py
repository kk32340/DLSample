import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import re

# ---- Tokenizer & Helper Functions ----
def tokenize(text):
    return re.findall(r'\b\w+\b', text.lower())

def bag_of_words(tokens, vocab):
    return np.array([1 if word in tokens else 0 for word in vocab], dtype=np.float32)

# ---- Sample Data ----
data = [
    ("Hi", "greeting"),
    ("Hello", "greeting"),
    ("Bye", "goodbye"),
    ("See you", "goodbye"),
    ("Thanks", "thanks"),
    ("Book flight", "booking"),
    ("I want to fly", "booking"),
]

# ---- Prepare Vocabulary and Labels ----
sentences, labels = zip(*data)
tokens = [token for s in sentences for token in tokenize(s)]
vocab = sorted(set(tokens))
tag_set = sorted(set(labels))
tag2idx = {tag: i for i, tag in enumerate(tag_set)}

X = [bag_of_words(tokenize(s), vocab) for s in sentences]
y = [tag2idx[label] for label in labels]

# ---- Convert to Tensors ----
X_train = torch.tensor(X, dtype=torch.float32)
y_train = torch.tensor(y, dtype=torch.long)

# ---- Model ----
class IntentNet(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.fc = nn.Linear(input_size, num_classes)
    
    def forward(self, x):
        return self.fc(x)

model = IntentNet(len(vocab), len(tag_set))
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# ---- Training ----
for epoch in range(200):
    outputs = model(X_train)
    loss = criterion(outputs, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 50 == 0:
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# ---- Prediction ----
def predict(sentence):
    with torch.no_grad():
        bow = bag_of_words(tokenize(sentence), vocab)
        output = model(torch.tensor(bow).unsqueeze(0))
        pred = torch.argmax(output)
        return tag_set[pred.item()]

# ---- Test ----
while True:
    text = input("You: ")
    if text.lower() == "quit":
        break
    intent = predict(text)
    print(f"Intent: {intent}")
