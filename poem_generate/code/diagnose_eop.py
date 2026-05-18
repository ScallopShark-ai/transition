"""
诊断: 模型是否真的学到了在『真实结束位置』预测 \<EOP\>。
对训练集中随机采样 N 首诗，找到每首诗里 \<EOP\> 的位置 p，
让模型 forward 到位置 p-1，看它在那一步给 \<EOP\> 多少概率 / 排名第几。
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import Config

CKPT = 'checkpoints/tang_99.pth'
N    = 500
SEED = 0

# --- 模型定义（避免 main.py 的循环导入）---
class PoetryModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=Config.num_layers)
        self.linear = nn.Linear(hidden_dim, vocab_size)
    def forward(self, x, hidden=None):
        seq_len, batch = x.size()
        if hidden is None:
            h0 = x.data.new(Config.num_layers, batch, self.hidden_dim).fill_(0).float()
            c0 = x.data.new(Config.num_layers, batch, self.hidden_dim).fill_(0).float()
        else:
            h0, c0 = hidden
        e = self.embeddings(x)
        out, hidden = self.lstm(e, (h0, c0))
        out = self.linear(out.view(seq_len*batch, -1))
        return out, hidden

# --- 加载数据和模型 ---
d = np.load('tang.npz', allow_pickle=True)
data = torch.from_numpy(d['data']).long()
ix2word = d['ix2word'].item()
word2ix = d['word2ix'].item()
PAD, EOP, START = word2ix['</s>'], word2ix['<EOP>'], word2ix['<START>']

device = torch.device('cuda')
model = PoetryModel(len(word2ix), Config.embedding_dim, Config.hidden_dim).to(device)
model.load_state_dict(torch.load(CKPT, map_location=device))
model.eval()

# --- 采样 ---
np.random.seed(SEED)
idx = np.random.choice(len(data), N, replace=False)
sample = data[idx]

ranks, probs, top3_records, full_examples = [], [], [], []

with torch.no_grad():
    for i in range(N):
        seq = sample[i]
        eop_pos = (seq == EOP).nonzero().squeeze(-1)
        if eop_pos.numel() == 0:
            continue
        p = eop_pos[0].item()
        if p == 0:
            continue
        inp = seq[:p].unsqueeze(1).to(device)        # (p, 1)
        out, _ = model(inp)                           # (p, vocab)
        probs_t = F.softmax(out[-1], dim=-1)           # 最后一步预测分布
        eop_p = probs_t[EOP].item()
        rank  = (probs_t > probs_t[EOP]).sum().item() + 1
        ranks.append(rank)
        probs.append(eop_p)
        top3_idx = torch.topk(probs_t, 3).indices.tolist()
        top3_chars = [ix2word[k] for k in top3_idx]
        top3_records.append((rank, top3_chars))
        if len(full_examples) < 6:
            start_idx = (seq == START).nonzero().squeeze(-1)
            poem_chars = []
            if start_idx.numel() > 0:
                s = start_idx[0].item()
                for j in range(s+1, p):
                    poem_chars.append(ix2word[seq[j].item()])
            full_examples.append({
                'poem': ''.join(poem_chars),
                'rank': rank,
                'prob': eop_p,
                'top3': top3_chars,
            })

ranks = np.array(ranks); probs = np.array(probs)

print(f'有效样本数: {len(ranks)} / {N}')
print(f'词表大小: {len(word2ix)}')
print()
print(f'== EOP 在真实结束位置的预测排名（越小越好，1 = 模型最该想到） ==')
def pct(c, t): return f'{c}/{t}  ({c/t*100:.1f}%)'
total = len(ranks)
print(f'  rank == 1 (top-1):      {pct((ranks == 1).sum(), total)}')
print(f'  rank ≤ 3:                {pct((ranks <= 3).sum(), total)}')
print(f'  rank ≤ 10:               {pct((ranks <= 10).sum(), total)}')
print(f'  rank ≤ 50:               {pct((ranks <= 50).sum(), total)}')
print(f'  rank > 50:               {pct((ranks > 50).sum(), total)}')
print(f'  rank > 500:              {pct((ranks > 500).sum(), total)}')
print(f'  rank 中位数: {int(np.median(ranks))},  均值: {ranks.mean():.1f}')
print()
print(f'== EOP 概率（softmax 后）==')
print(f'  中位数: {np.median(probs)*100:.4f}%')
print(f'  均值:   {probs.mean()*100:.4f}%')
print(f'  最大:   {probs.max()*100:.4f}%')
print(f'  最小:   {probs.min()*100:.6f}%')
print()
print(f'== 6 条样本细节 ==')
for s in full_examples:
    print(f"  诗: {s['poem'][:60]}{'...' if len(s['poem']) > 60 else ''}")
    print(f"     EOP 排名: {s['rank']:>5},  EOP 概率: {s['prob']*100:.4f}%")
    print(f"     模型在该位置实际 top-3: {s['top3']}")
    print()
