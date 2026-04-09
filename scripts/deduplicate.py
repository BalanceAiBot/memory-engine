#!/usr/bin/env python3
"""
记忆库去重脚本 (Deduplication Script)
使用 Embedding 语义相似度检测重复条目并合并。
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.engine import MemoryEngine

def deduplicate_memory(engine: MemoryEngine, threshold: float = 0.85):
    """
    检测并删除重复记忆。
    策略：保留文本更长（信息量更大）的那一条。
    """
    print("🔍 开始扫描重复记忆...")
    
    # 1. 获取所有记忆
    all_chunks = engine.store.get_all_chunks()
    print(f"📊 当前总数: {len(all_chunks)} 条")

    if len(all_chunks) < 2:
        print("✅ 记忆太少，无需去重。")
        return 0

    # 2. 计算两两相似度矩阵
    # 为了效率，先收集所有文本
    texts = [c['text'] for c in all_chunks]
    
    print("🧠 计算语义向量...")
    vectors = engine.embedder.encode(texts)
    
    import numpy as np
    # 计算余弦相似度矩阵 (向量化运算)
    norm_vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    similarity_matrix = np.dot(norm_vectors, norm_vectors.T)

    # 3. 识别重复对
    to_remove_ids = set()
    
    # 只遍历上三角矩阵 (i < j)
    for i in range(len(all_chunks)):
        if all_chunks[i]['chunk_id'] in to_remove_ids:
            continue
            
        for j in range(i + 1, len(all_chunks)):
            if all_chunks[j]['chunk_id'] in to_remove_ids:
                continue
                
            sim = similarity_matrix[i][j]
            
            if sim >= threshold:
                # 发现重复！
                len_i = len(texts[i])
                len_j = len(texts[j])
                
                # 策略：保留较长的那条
                if len_i >= len_j:
                    victim = all_chunks[j]
                else:
                    victim = all_chunks[i]
                
                print(f"  ⚠️ 发现重复 (相似度 {sim:.2f}):")
                print(f"    保留: {texts[i if len_i >= len_j else j][:40]}...")
                print(f"    删除: {texts[j if len_i >= len_j else i][:40]}...")
                
                to_remove_ids.add(victim['chunk_id'])

    # 4. 执行删除
    if not to_remove_ids:
        print("✅ 没有发现重复项。")
        return 0

    print(f"\n🗑️ 准备删除 {len(to_remove_ids)} 条重复记录...")
    
    # 注意：engine.delete 需要处理 FAISS 索引
    # 由于我们要批量删除，直接 rebuild 索引更干净
    chunk_ids_to_remove = list(to_remove_ids)
    
    # 先从 DB 删除
    for cid in chunk_ids_to_remove:
        engine.store.delete_chunk(cid)
        
    # 重建 FAISS 索引
    engine.rebuild_index()
    
    print(f"✅ 去重完成！剩余记忆: {len(all_chunks) - len(to_remove_ids)} 条")
    return len(to_remove_ids)

if __name__ == "__main__":
    engine = MemoryEngine(db_path="~/.hermes/memories/memory.db")
    deduplicate_memory(engine)
