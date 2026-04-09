#!/usr/bin/env python3
"""
记忆数据清洗与优化脚本 (Data Hygiene)
目标：去除 Markdown 噪音，修复断裂句子，提升向量召回精度。
"""
import re
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.engine import MemoryEngine

def clean_text(text):
    """清理单条文本中的噪音。"""
    if not text:
        return ""
    
    # 1. 去除 Markdown 格式
    text = re.sub(r'#{1,6}\s*', '', text)          # 标题符号
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)     # 加粗
    text = re.sub(r'\*(.*?)\*', r'\1', text)         # 斜体
    text = re.sub(r'---+', '', text)                 # 分割线
    text = re.sub(r'§', ' ', text)                   # 分隔符替换为空格
    text = re.sub(r'`', '', text)                    # 代码块符号
    text = re.sub(r'\\', '', text)                   # 转义符

    # 2. 清理表格 (提取关键内容)
    # 简单的表格行通常是 | xxx | xxx |
    if text.count('|') > 2:
        # 移除首尾管道符和空格
        text = text.strip('| ').replace('|', ', ')
        # 去除连续的逗号
        text = re.sub(r',\s*,', ',', text)

    # 3. 处理列表符号
    text = re.sub(r'^\s*[-*+]\s+', '', text, flags=re.MULTILINE)

    # 4. 清理多余空格和换行
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def run_cleaning():
    engine = MemoryEngine(db_path="~/.hermes/memories/memory.db")
    all_chunks = engine.store.get_all_chunks()
    
    print(f"🧹 开始清洗 {len(all_chunks)} 条记忆...")
    
    cleaned_count = 0
    for chunk in all_chunks:
        original = chunk['text']
        cleaned = clean_text(original)
        
        if original != cleaned:
            engine.store.update_chunk(chunk['chunk_id'], text=cleaned)
            cleaned_count += 1
            # 打印前几个看看效果
            if cleaned_count <= 3:
                print(f"  - 原始: {original[:50]}...")
                print(f"  - 清洗: {cleaned[:50]}...")

    print(f"✅ 清洗完成：优化了 {cleaned_count} 条。")
    
    # 强制重建向量索引以应用新文本
    print("🔄 重建向量索引...")
    engine.rebuild_index()
    print("✅ 索引重建完成。")

if __name__ == "__main__":
    run_cleaning()
