import torch
from torch.utils.data import DataLoader, TensorDataset

import torch.nn as nn

resume_dir = '/content/drive/MyDrive/comparison/result/ckpt_iter_900_acc.pt'

n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.2
data_format='plain'
operator=','
dataset = 'bal'
batch_size = 512
block_size = 16 # context of up to 256 previous characters
use_flash = True
bias = False



# Load model and checkpoint
checkpoint = torch.load(resume_dir, map_location=device)
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=None, dropout=dropout, use_flash=use_flash)

gptconf = GPTConfig(**model_args)
model = GPT(gptconf)
model.to(device)





def num_to_embeddings(num, max_digits=3):
    """Convert integer to concatenated digit embeddings."""
    digits = [int(x) for x in f"{num:0{max_digits}d}"]  # e.g., 7 -> 007
    return torch.cat([digit_embeddings[d] for d in digits])


# Create dataset
def make_dataset(n_samples=5000):
    X, y = [], []
    for _ in range(n_samples):
        a = random.randint(0, 999)
        b = random.randint(0, 999)
        # you can allow negatives or not
        result = a - b  
        emb_a = num_to_embeddings(a, 3)
        emb_b = num_to_embeddings(b, 3)
        x = torch.cat([emb_a, emb_b])   # input vector 6d
        X.append(x)
        y.append(result)
    return torch.stack(X), torch.tensor(y, dtype=torch.float32).unsqueeze(1)


X, y = make_dataset(10000)
train_X, test_X = X[:8000], X[8000:]
train_y, test_y = y[:8000], y[8000:]

probe = nn.Linear(train_X.shape[1], 1)
criterion = nn.MSELoss()
optimizer = optim.Adam(probe.parameters(), lr=1e-3)

for epoch in range(2000):
    optimizer.zero_grad()
    out = probe(train_X)
    loss = criterion(out, train_y)
    loss.backward()
    optimizer.step()
    if epoch % 200 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# Evaluate
with torch.no_grad():
    preds = probe(test_X).round().squeeze()
    acc = (preds == test_y.squeeze()).float().mean()
    print("Exact subtraction accuracy:", acc.item())