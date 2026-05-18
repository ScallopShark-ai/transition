"""
把 chinese-poetry/全唐诗/poet.tang.*.json 里的繁体字转成简体，
另存到 data/chinese-poetry/全唐诗_简体/ 下，保持原文件名和 json 结构不变。
"""
import glob, json, os
from opencc import OpenCC

SRC_DIR = 'data/chinese-poetry/全唐诗'
DST_DIR = 'data/chinese-poetry/全唐诗_简体'
PATTERN = 'poet.tang.*.json'

cc = OpenCC('t2s')
os.makedirs(DST_DIR, exist_ok=True)

files = sorted(glob.glob(f'{SRC_DIR}/{PATTERN}'))
print(f'匹配 {len(files)} 个文件')

for i, fp in enumerate(files):
    with open(fp, 'r', encoding='utf-8') as f:
        arr = json.load(f)
    for poem in arr:
        if 'title' in poem:
            poem['title']  = cc.convert(poem['title'])
        if 'author' in poem:
            poem['author'] = cc.convert(poem['author'])
        if 'paragraphs' in poem:
            poem['paragraphs'] = [cc.convert(s) for s in poem['paragraphs']]
    dst = os.path.join(DST_DIR, os.path.basename(fp))
    with open(dst, 'w', encoding='utf-8') as f:
        json.dump(arr, f, ensure_ascii=False, indent=2)
    if (i+1) % 10 == 0 or i+1 == len(files):
        print(f'  {i+1}/{len(files)} 完成')

print(f'输出目录: {DST_DIR}')
