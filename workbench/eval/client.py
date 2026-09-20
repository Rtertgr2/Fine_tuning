"""HTTP client for llama-server (OpenAI-compatible)."""
from __future__ import annotations
import json, time, urllib.request, urllib.error
from typing import Optional


class LlamaClient:
    def __init__(self, base_url: str = "http://127.0.0.1:8080"):
        self.base_url = base_url.rstrip("/")
    
    def chat(self, messages: list, tools: Optional[list] = None,
             temperature: float = 0.0, max_tokens: int = 2048,
             timeout: int = 120) -> dict:
        """Call /v1/chat/completions and return parsed result."""
        body = {
            "model": "default",
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if tools:
            body["tools"] = tools
        
        req = urllib.request.Request(
            self.base_url + "/v1/chat/completions",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        start = time.time()
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read().decode()
                result = json.loads(raw)
        except urllib.error.HTTPError as e:
            return {"error": f"HTTP {e.code}: {e.read().decode()[:200]}", "elapsed": time.time() - start}
        except Exception as e:
            return {"error": str(e), "elapsed": time.time() - start}
        
        elapsed = time.time() - start
        usage = result.get("usage", {})
        choice = result.get("choices", [{}])[0]
        msg = choice.get("message", {})
        
        return {
            "content": msg.get("content", ""),
            "tool_calls": msg.get("tool_calls", []),
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
            "elapsed": elapsed,
            "timings": result.get("timings", {}),
        }
    
    def tokenize(self, text: str) -> list:
        """Call /tokenize to get token ids."""
        body = {"content": text}
        req = urllib.request.Request(
            self.base_url + "/tokenize",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                return json.loads(resp.read().decode()).get("tokens", [])
        except Exception:
            return []
    
    def health(self) -> bool:
        try:
            req = urllib.request.Request(self.base_url + "/health")
            with urllib.request.urlopen(req, timeout=5) as resp:
                return resp.status == 200
        except Exception:
            return False
