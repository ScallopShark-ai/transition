import re
import os
import matplotlib.pyplot as plt

LOSS_FILE = 'result.txt'
OUT_DIR   = '/data1/daishizhe/workspace/poem_generate/code/report/exp-002'
OUT_PNG   = os.path.join(OUT_DIR, 'loss_curve.png')

os.makedirs(OUT_DIR, exist_ok=True)

losses = []
with open(LOSS_FILE, 'r', encoding='utf-8') as f:
    for line in f:
        m = re.match(r'训练损失为([\d.]+)', line)
        if m:
            losses.append(float(m.group(1)))

epochs = list(range(len(losses)))

plt.figure(figsize=(8, 5))
plt.plot(epochs, losses, marker='o', linewidth=1.5)
plt.xlabel('Epoch')
plt.ylabel('Training Loss')
plt.title(f'Loss Curve ({len(losses)} epochs)')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUT_PNG, dpi=120)
print(f'共 {len(losses)} 个 epoch')
if losses:
    print(f'最小 loss: {min(losses):.4f} (epoch {losses.index(min(losses))})')
    print(f'最终 loss: {losses[-1]:.4f}')
print(f'图已保存: {OUT_PNG}')
