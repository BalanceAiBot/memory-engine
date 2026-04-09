#!/usr/bin/env python3
"""
轻量级 HTTP 服务，用于加速 Memory Engine 检索。
保持模型常驻内存，避免每次请求重新加载。
"""
import sys
import os
import json
import urllib.parse
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

# 🔥 关键：开启离线模式，防止网络不稳定导致模型加载崩溃
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

# 必须在导入 torch 之前设置
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

sys.path.insert(0, os.path.dirname(__file__))

PORT = 54321

class MemoryHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        
        if parsed.path == '/search':
            # print(f"[SERVER] Raw Path: {self.path}", file=sys.stderr)
            query = parse_qs(parsed.query).get('q', [''])[0]
            # 修复可能的 URL 编码问题 (Mojibake Fix)
            try:
                if len(query) > 4 and '\xe4' in query:
                    query = query.encode('latin-1').decode('utf-8')
            except:
                pass
            # print(f"[SERVER] Fixed Query: {query}", file=sys.stderr)
            # print(f"[SERVER] Decoded Query: {query}", file=sys.stderr)
            
            top_k = int(parse_qs(parsed.query).get('k', ['5'])[0])
            
            if not query:
                self._send_json({"error": "Missing 'q' parameter"}, 400)
                return

            try:
                # 使用混合检索
                # print(f"[SERVER] Query: {query}")
                results = engine.query(query, top_k=top_k)
                
                response = {
                    "success": True,
                    "query": query,
                    "result_count": len(results),
                    "results": [
                        {
                            "rank": i + 1,
                            "score": round(r["score"], 4),
                            "category": r.get("category", ""),
                            "text": r["text"]
                        } for i, r in enumerate(results)
                    ]
                }
                self._send_json(response)
                
            except Exception as e:
                self._send_json({"success": False, "error": str(e)}, 500)
        
        elif parsed.path == '/health':
            self._send_json({"status": "ok"})
            
        else:
            self._send_json({"error": "Not found"}, 404)

    def _send_json(self, data, status=200):
        self.send_response(status)
        self.send_header("Content-type", "application/json; charset=utf-8")
        self.end_headers()
        self.wfile.write(json.dumps(data, ensure_ascii=False).encode("utf-8"))

    def log_message(self, format, *args):
        # 屏蔽默认日志，保持控制台干净
        pass

if __name__ == "__main__":
    print("🚀 Loading Memory Engine...")
    try:
        from src.engine import MemoryEngine
        engine = MemoryEngine(db_path=os.path.expanduser("~/.hermes/memories/memory.db"))
        
        # 预热: 初始化 BM25 索引
        engine.retriever.update_bm25_index(engine.store.get_all_chunks())
        print(f"✅ BM25 Index built ({len(engine.store.get_all_chunks())} chunks)")
        
        # 预热模型
        engine.query("warmup", top_k=1)
        print("✅ Memory Engine loaded and warmed up.")
    except Exception as e:
        print(f"❌ Failed to load engine: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    class ReuseAddrServer(HTTPServer):
        allow_reuse_address = True

    print(f"👂 Server running on http://127.0.0.1:{PORT}")
    try:
        server = ReuseAddrServer(("127.0.0.1", PORT), MemoryHandler)
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()
