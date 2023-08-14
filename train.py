import torch
import torch.nn as nn
from torch.nn import functional as F 

batch_size = 32
block_size = 8
max_iters = 5000
eval_iterval = 300
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embed = 32

#---------------
torch.manual_seed(1337)


with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()
    
chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

   
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    # generating `batch size` number of indices in the range of our data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(split)
            logits, loss = model(x, y)
            losses[k] = loss.item()
            
        out[split] = losses.mean()
        
    model.train()
    return out
        
        
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query =  nn.Linear(n_embed, head_size, bias=False)
        self.value =  nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        
    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * C**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] ==0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        v = self.value(x)
        out = wei @ v
        return out
    
    
class MultiHeadAttention(nn.Module):
    """multiple heads of self-attention in parallel"""
    
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        
    def forward(self, x):
        return torch.cat([h(x) for h in self.heads], dim=-1) # Concatinating over channel dimension
    
    
class FeedForward(nn.Module):
    ''' a simple linear layer followd by a non-linearity'''
    
    """ before implementing this layer nodes where talking to each other, but they didn't have time to learn the 
    the information that they have received through the comunication."""
    
    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, n_embed),
            nn.ReLU()
        )
        
    def forward(self, x):
        return self.net(x)
        

class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        # In most cases not only we need the embedding of tokens but also we need the embedding of 
        # their positions. So, here we will add the positional embeddings
        self.postion_embedding_table = nn.Embedding(block_size, n_embed)
        self.sa_heads = MultiHeadAttention(4, n_embed//4) # i.e. 4 heads of 8-dimensional self-attention (very similar to group convolution)
        self.ffwd = FeedForward(n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)
        
    def forward(self, idx, targets=None):
        B, T = idx.shape
        # idx and targets are both (B,T) tensor of integers
        # Instead of having vocab by vocab size we will use an embedding. So, we cannot compute
        # the logits directly and we need to add a linear layer (See the constructor of the model)
        token_embedding = self.token_embedding_table(idx) # (B,T,C_{embedding})
        # In next line torch.arrange will generate integers from 0 to T-1
        pos_embedding = self.postion_embedding_table(torch.arange(T, device=device)) #(T, C)
        # the addition below will use broadcasting in pytorch
        x = token_embedding + pos_embedding
        x = self.sa_heads(x)
        x = self.ffwd(x)
        logits = self.lm_head(x) #(B,T,C_{vocab_size})
        

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            
            
            # Since we have positional embedding we can not allow a data larger than
            # the block_size. If we don't do this then the positional embedding table will go out of scope
            idx_cond = idx[:, -block_size:]
            
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx


model = BigramLanguageModel()
m = model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters): # increase number of steps for good results...

    if iter % eval_iterval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss: {losses['train']:.4f} and eval loss: {losses['val']:.4f}")
    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
   
context = torch.zeros((1, 1), dtype=torch.long, device=device) 
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))