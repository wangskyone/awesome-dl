import torch
from torch import nn
import torch.nn.functional as F
import math
import random
import numpy as np

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len, device):
        super().__init__()

        self.encoding = torch.zeros(max_len, d_model, device=device).requires_grad_(False)

        pos = torch.arange(0, max_len, device=device).float().unsqueeze(1)

        _2i = torch.arange(0, d_model, step=2, device=device).float()


        self.encoding[:,0::2] = torch.sin(pos / (10000 ** (_2i / d_model)))
        self.encoding[:,1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))

    
    def forward(self, x):

        batch_size, seq_len = x.size()
        
        return self.encoding[:seq_len, :]
    

class TokenEmbedding(nn.Embedding):

    def __init__(self, vocab_size, d_model):
        super().__init__(vocab_size, d_model, padding_idx=1)



class TransformerEncoding(nn.Module):

    def __init__(self, vocab_size, d_model, max_len, drop_prob, device) -> None:
        super().__init__()

        self.tok_emb = TokenEmbedding(vocab_size, d_model)
        self.pos_emb = PositionalEncoding(d_model, max_len, device)
        self.drop_out = nn.Dropout(drop_prob)

    def forward(self, x):

        tok_emb = self.tok_emb(x)
        pos_emb = self.pos_emb(x)

        return self.drop_out(tok_emb + pos_emb)

    

class ScaleDotAttention(nn.Module):

    def __init__(self,):
        super().__init__()


    def forward(self, q, k, v, mask=None, eps=1e-12):

        batch_size, head, length, dim = q.size()

        k_t = k.transpose(2, 3)
        score = F.softmax(q @ k_t, dim=-1) / math.sqrt(dim)

        if mask is not None:
            score = score.masked_fill(mask==0, -10000)

        score = F.softmax(score, dim=-1)

        v = score @ v

        return v, score
    

class MultiHeadAttention(nn.Module):

    def __init__(self,d_model, n_head):
        super().__init__()

        self.n_head = n_head
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_concat = nn.Linear(d_model, d_model)
        self.attention = ScaleDotAttention()

    
    def forward(self, q, k, v, mask=None):

        b, s, d= q.size()

        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)

        #1.split

        d_tensor= d // self.n_heads

        q, k, v = q.view(b, s, self.n_head, d_tensor).transpose(1 , 2), k.view(b, s, self.n_head, d_tensor).transpose(1 , 2), v.view(b, s, self.n_head, d_tensor).transpose(1 , 2)

        #2.scaledot

        out, score = self.attention(q, k, v, mask)


        #3.concat


        out = out.transpose(1 , 2).contiguous().view(b, s, d)
        out = self.w_concat(out)

        #4.out

        return out
    

class LayerNorm(nn.Module):

    def __init__(self, d_model, eps=1e-12) -> None:
        super().__init__()

        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):

        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, unbiased=False, keepdim=True)

        out = (x - mean) / torch.sqrt(var + self.eps)
        out = self.gamma * out + self.beta


        return out

# b s d
class BatchNorm(nn.Module):

    def __init__(self,s , d, eps=1e-12) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(1, s ,d))
        self.beta = nn.Parameter(torch.zeros(1, s, d))
        self.eps = eps

    def forward(self, x):

        mean = x.mean(0, keepdim=True)
        var = x.var(0, unbiased=False, keepdim=True)

        out = (x - mean) / torch.sqrt(var + self.eps)
        out = self.gamma * out + self.beta


        return out


class PositionwiseFeedForward(nn.Module):

    def __init__(self, d_model, hidden, drop_prob=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, hidden)
        self.linear2 = nn.Linear(hidden, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=drop_prob)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class EncoderLayer(nn.Module):

    def __init__(self, d_model, n_head, hidden, drop_prob=0.1) -> None:
        super().__init__()

        self.attention = MultiHeadAttention(d_model, n_head)
        self.norm1 = LayerNorm(d_model)
        self.dropout1 = nn.Dropout(drop_prob)

        self.ffn = PositionwiseFeedForward(d_model, hidden, drop_prob)
        self.norm2 = LayerNorm(d_model)
        self.dropout2 = nn.Dropout(drop_prob)


    
    def forwrd(self, x, srcmask):

        _x = x
        x = self.attention(q=x, k=x, v=x, mask=srcmask)
        x = self.dropout1(x)
        x = self.norm1(x + _x)


        _x = x
        x = self.ffn(x)
        x = self.dropout2(x)
        x = self.norm2(x + _x)

        return x
    

class Encoder(nn.Module):

    def __init__(self, d_model, hidden , n_head, max_len, vocab_size, drop_prob, n_layers, device) -> None:
        super().__init__()

        self.emb = TransformerEncoding(vocab_size=vocab_size, d_model=d_model, max_len=max_len, drop_prob=drop_prob, device=device)

        self.layers = nn.ModuleList([
            EncoderLayer(d_model=d_model, n_head=n_head, hidden=hidden, drop_prob=drop_prob)
            for _ in range(n_layers)
        ])

    def forward(self, x, src_mask):

        x = self.emb(x)

        for layer in self.layers:
            x = layer(x, src_mask)

        return x
    

class DecoderLayer(nn.Module):

    def __init__(self,d_model, hidden, n_head, drop_prob) -> None:
        super().__init__()
        
        self.self_attention = MultiHeadAttention(d_model=d_model, n_head=n_head)
        self.norm1 = LayerNorm(d_model)
        self.dropout1 = nn.Dropout(drop_prob)


        self.enc_dec_attention = MultiHeadAttention(d_model=d_model, n_head=n_head)
        self.norm2 = LayerNorm(d_model)
        self.dropout2 = nn.Dropout(drop_prob)


        self.ffn = PositionwiseFeedForward(d_model=d_model, hidden=hidden, drop_prob=drop_prob)
        self.norm3 = LayerNorm(d_model)
        self.dropout3 = nn.Dropout(drop_prob)


    
    def forward(self, dec, enc, trg_mask, src_mask):

        _x = dec
        x = self.self_attention(q=dec, k=dec, v=dec, mask=trg_mask)

        x = self.dropout1(x)
        x = self.norm1(x + _x)

        if enc is not None:

            _x = x
            x = self.enc_dec_attention(q=x, k=enc, v=enc, mask=src_mask)


            x = self.dropout2(x)
            x = self.norm2(x + _x)


        _x = x
        x = self.ffn(x)

        x = self.dropout3(x)
        x = self.norm3(_x + x)

        return x
    
class Decoder(nn.Module):
    
    def __init__(self, d_model, hidden , n_head, max_len, vocab_size, drop_prob, n_layers, device):
        super().__init__()

        self.emb = TransformerEncoding(vocab_size=vocab_size, d_model=d_model, max_len=max_len, drop_prob=drop_prob, device=device)

        self.layers = nn.ModuleList([
            DecoderLayer(d_model=d_model, hidden=hidden, n_head=n_head, drop_prob=drop_prob)
            for _ in range(n_layers)
        ])

        self.de_vocab = nn.Linear(d_model, vocab_size)


    def forward(self, trg, src, trg_mask, src_mask):
        
        trg = self.emb(trg)

        for layer in self.layers:
            trg = layer(trg, src, trg_mask, src_mask)

        output = self.de_vocab(trg)

        return output


class Transformer(nn.Module):

    def __init__(self, src_pad_idx, trg_pad_idx, trg_sos_idx, vocab_size, d_model, n_head, max_len,
                 hidden, n_layers, drop_prob, device):
        super().__init__()
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.trg_sos_idx = trg_sos_idx
        self.device = device
        self.encoder = Encoder(d_model=d_model,
                               n_head=n_head,
                               max_len=max_len,
                               hidden=hidden,
                               vocab_size=vocab_size,
                               drop_prob=drop_prob,
                               n_layers=n_layers,
                               device=device)

        self.decoder = Decoder(d_model=d_model,
                               n_head=n_head,
                               max_len=max_len,
                               hidden=hidden,
                               vocab_size=vocab_size,
                               drop_prob=drop_prob,
                               n_layers=n_layers,
                               device=device)

    def forward(self, src, trg):
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        enc_src = self.encoder(src, src_mask)
        output = self.decoder(trg, enc_src, trg_mask, src_mask)
        return output

    def make_src_mask(self, src):
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask

    def make_trg_mask(self, trg):
        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(3)
        trg_len = trg.shape[1]
        trg_sub_mask = torch.tril(torch.ones(trg_len, trg_len)).type(torch.ByteTensor).to(self.device)
        trg_mask = trg_pad_mask & trg_sub_mask
        return trg_mask


if __name__ == "__main__":
    
    model = Transformer(src_pad_idx=1,trg_pad_idx=1,trg_sos_idx=0,vocab_size=10000,d_model=512,n_head=8,hidden=2048,n_layers=6,drop_prob=0.1, max_len=100,device="cpu")
    print(model)