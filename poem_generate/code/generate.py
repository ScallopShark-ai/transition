import torch as t
import numpy as np
from torch.utils.data import DataLoader
from torch import optim
from torch import nn
from model import *
from torchnet import meter
import tqdm
from config import *
from test import *


# ============ 实验 3 推理改造超参（在此调） ============
TOP_P            = 0.9     # nucleus 采样阈值（保留累计概率 >= p 的最小集合）
TEMPERATURE      = 1.0     # softmax 温度，>1 更随机，<1 更尖锐
REPEAT_NGRAM     = 4       # 重复检测的 n-gram 大小
REPEAT_WINDOW    = 30      # 在最近多少 token 内查 n-gram
MAX_PERIOD_COUNT = 8       # 累计 N 个 "。" 就停（覆盖绝句 4 + 律诗 8）
DEBUG            = True    # 打印停止原因，便于调试

PUNCT_END = {'。', '，', '?', '！', '<START>'}


# -------------------------- 采样 --------------------------
def nucleus_sample(logits, p=TOP_P, temperature=TEMPERATURE):
    """top-p (nucleus) 采样：在概率累积到 p 的最小 token 集合里按概率重新采样。"""
    if temperature != 1.0:
        logits = logits / temperature
    probs = t.softmax(logits, dim=-1)
    sorted_probs, sorted_idx = t.sort(probs, descending=True)
    cum = t.cumsum(sorted_probs, dim=-1)
    mask = cum >= p
    cutoff = mask.nonzero()[0].item() if mask.any() else len(sorted_probs) - 1
    cand = sorted_probs[:cutoff + 1].clone()
    cand = cand / cand.sum()
    chosen = t.multinomial(cand, 1).item()
    return sorted_idx[chosen].item()


def has_ngram_repeat(tok_ids, n=REPEAT_NGRAM, window=REPEAT_WINDOW):
    """判断 tok_ids 末尾的 n-gram 是否在最近 window 个 token 之内已经出现过。"""
    if len(tok_ids) < 2 * n:
        return False
    last = tuple(tok_ids[-n:])
    start = max(0, len(tok_ids) - window)
    earlier = tok_ids[start:-n]
    for i in range(len(earlier) - n + 1):
        if tuple(earlier[i:i + n]) == last:
            return True
    return False


# -------------------------- 首句生成 --------------------------
def generate(model, start_words, ix2word, word2ix, prefix_words=None):
    results = list(start_words)
    start_words_len = len(start_words)
    input = t.Tensor([word2ix['<START>']]).view(1, 1).long()
    if Config.use_gpu:
        input = input.cuda()
    hidden = None

    if prefix_words:
        for word in prefix_words:
            output, hidden = model(input, hidden)
            input = input.data.new([word2ix[word]]).view(1, 1)

    # 跟踪已生成的字 id，用于 n-gram 检测；首句已知字也算进去
    generated_ids = [word2ix[c] for c in start_words if c in word2ix]
    period_count  = 0
    stop_reason   = 'max_gen_len'

    for i in range(Config.max_gen_len):
        output, hidden = model(input, hidden)

        if i < start_words_len:
            # 还在喂用户给的首句字符
            w = results[i]
            input = input.data.new([word2ix[w]]).view(1, 1)
            continue

        # 真正开始采样
        top_index = nucleus_sample(output.data[0])
        w = ix2word[top_index]
        results.append(w)
        generated_ids.append(top_index)
        input = input.data.new([top_index]).view(1, 1)

        # 兜底 1：模型自己发射 EOP
        if w == '<EOP>':
            del results[-1]
            stop_reason = '<EOP>'
            break

        # 兜底 2：n-gram 重复
        if has_ngram_repeat(generated_ids):
            stop_reason = f'repeat-{REPEAT_NGRAM}gram'
            break

        # 兜底 3：累计 "。" 数
        if w == '。':
            period_count += 1
            if period_count >= MAX_PERIOD_COUNT:
                stop_reason = f'{MAX_PERIOD_COUNT}-periods'
                break

    if DEBUG:
        print(f'[generate] len={len(results)}, stop_reason={stop_reason}')
    return results


# -------------------------- 藏头生成 --------------------------
def gen_acrostic(model, start_words, ix2word, word2ix, prefix_words=None):
    result = []
    start_words_len = len(start_words)
    input = (t.Tensor([word2ix['<START>']]).view(1, 1).long())
    if Config.use_gpu:
        input = input.cuda()
    index = 0
    pre_word = '<START>'
    hidden = None

    if prefix_words:
        for word in prefix_words:
            output, hidden = model(input, hidden)
            input = (input.data.new([word2ix[word]])).view(1, 1)

    generated_ids = []     # 用于句内 n-gram 防卡死
    stop_reason = 'max_gen_len'

    for i in range(Config.max_gen_len):
        output, hidden = model(input, hidden)
        top_index = nucleus_sample(output.data[0])
        w = ix2word[top_index]

        if pre_word in PUNCT_END:
            # 上一步刚收完一句，下一字必须是用户给的藏头字
            if index == start_words_len:
                stop_reason = 'all-acrostic-chars-used'
                break
            w = start_words[index]
            index += 1
            input = (input.data.new([word2ix[w]])).view(1, 1)
            generated_ids = []          # 进入新句，重置句内 n-gram 缓冲
        else:
            input = (input.data.new([top_index])).view(1, 1)
            generated_ids.append(top_index)
            # 句内 n-gram 重复 → 强制收句（发射 "。"）让外层的 "逢句末喂下一藏头字" 逻辑接管
            if has_ngram_repeat(generated_ids):
                w = '。'
                input = (input.data.new([word2ix['。']])).view(1, 1)

        result.append(w)
        pre_word = w

    if DEBUG:
        print(f'[gen_acrostic] len={len(result)}, stop_reason={stop_reason}')
    return result
