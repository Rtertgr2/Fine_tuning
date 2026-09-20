#!/usr/bin/env python3
"""Generate frozen test suites for Eval Harness (T4.2, T4.3, T4.4).

Produces cases.jsonl + manifest.json for each suite.
All suites use distinct group prefixes from the training data to prevent leakage.
"""
import json, hashlib
from pathlib import Path
from datetime import datetime, timezone

BASE = Path(__file__).resolve().parent.parent
SUITES = BASE / "eval/suites"

def manifest(suite_name, version, cases_path, extra=None):
    data = [json.loads(line) for line in Path(cases_path).read_text().splitlines() if line.strip()]
    content = Path(cases_path).read_bytes()
    m = {
        "suite": suite_name,
        "version": version,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "cases_count": len(data),
        "cases_sha256": hashlib.sha256(content).hexdigest(),
        "frozen": True,
        "overlap_check": {"method": "group_prefix", "train_groups": []},
    }
    if extra:
        m.update(extra)
    return m

def write_suite(name, version, cases, extra=None):
    d = SUITES / name / version
    d.mkdir(parents=True, exist_ok=True)
    cp = d / "cases.jsonl"
    with open(cp, "w") as f:
        for c in cases:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    m = manifest(name, version, cp, extra)
    with open(d / "manifest.json", "w") as f:
        json.dump(m, f, indent=2, ensure_ascii=False)
    print(f"{name}/{version}: {len(cases)} cases, sha256={m['cases_sha256'][:16]}...")

# ── TOOL_SYNTAX suite ──────────────────────────────────────────
def gen_tool_syntax():
    cases = []
    # Simple single tool calls
    tasks = [
        ("Read the file src/main.py", "read_file", {"path": "src/main.py"}),
        ("Write a hello world to hello.py", "write_file", {"path": "hello.py", "content": "print('hello')\n"}),
        ("Checkout branch feature/login", "git_checkout", {"repo": ".", "branch": "feature/login"}),
        ("Read config.yaml", "read_file", {"path": "config.yaml"}),
        ("Create a new utils.py", "write_file", {"path": "utils.py", "content": "def add(a, b): return a + b\n"}),
    ]
    for i, (user, tool, args) in enumerate(tasks):
        cases.append({
            "id": f"tool_syntax_{i+1:03d}",
            "group": f"eval/tool_syntax/batch1/task{i+1}",
            "prompt": user,
            "expected_tool": tool,
            "expected_args": args,
        })
    
    # Multiple tool calls in sequence
    for i in range(10):
        cases.append({
            "id": f"tool_syntax_{len(cases)+1:03d}",
            "group": f"eval/tool_syntax/batch2/seq{i}",
            "prompt": f"Read file_a_{i}.txt and write the result to file_b_{i}.txt",
            "expected_tool": "read_file",
            "expected_args": {"path": f"file_a_{i}.txt"},
            "follow_up": {"tool": "write_file", "args": {"path": f"file_b_{i}.txt"}},
        })
    
    # Varied prompts
    verbs = ["Show me", "Open", "Load", "Fetch", "Get", "Display", "Print", "Read"]
    for i, v in enumerate(verbs):
        cases.append({
            "id": f"tool_syntax_{len(cases)+1:03d}",
            "group": f"eval/tool_syntax/batch3/prompt{i}",
            "prompt": f"{v} the contents of log/app.log",
            "expected_tool": "read_file",
            "expected_args": {"path": "log/app.log"},
        })
    
    while len(cases) < 100:
        idx = len(cases) - 43
        cases.append({
            "id": f"tool_syntax_{len(cases)+1:03d}",
            "group": f"eval/tool_syntax/batch4/pad{idx}",
            "prompt": f"Read the source code from module_{idx}.py",
            "expected_tool": "read_file",
            "expected_args": {"path": f"module_{idx}.py"},
        })
    return cases

# ── ANTI_LOOP suite ────────────────────────────────────────────
def gen_anti_loop():
    cases = []
    errors = [
        ("File not found: src/missing.py", "read_file", {"path": "src/missing.py"}),
        ("Permission denied: /etc/shadow", "read_file", {"path": "/etc/shadow"}),
        ("No such file or directory: build/output.js", "read_file", {"path": "build/output.js"}),
        ("Branch 'nonexistent' not found", "git_checkout", {"repo": ".", "branch": "nonexistent"}),
        ("Syntax error in code block", "write_file", {"path": "broken.py", "content": "def foo("}),
        ("Read-only file system", "write_file", {"path": "/readonly/file.py", "content": "x=1\n"}),
    ]
    for i, (err_msg, tool, args) in enumerate(errors):
        cases.append({
            "id": f"anti_loop_{i+1:03d}",
            "group": f"eval/loop/errors/error{i}",
            "prompt": f"Try to call {tool} but get error: {err_msg}",
            "inject_error": err_msg,
            "must_differ": True,
            "must_not_repeat_same_call": True,
        })
    
    while len(cases) < 40:
        idx = len(cases)
        cases.append({
            "id": f"anti_loop_{len(cases)+1:03d}",
            "group": f"eval/loop/errors/batch2/e{idx}",
            "prompt": f"Simulated error scenario #{idx}",
            "inject_error": f"Simulated failure #{idx}",
            "must_differ": True,
        })
    return cases

# ── SECURITY VULN suite ────────────────────────────────────────
def gen_security_vuln():
    vulns = [
        ("sql_injection", "user.py", 3,
         "def get_user(db, uid):\n     return db.execute(f'SELECT * FROM users WHERE id={uid}').fetchone()\n",
         "SQL injection via f-string interpolation"),
        ("xss", "handler.py", 2,
         "def render(request):\n     return '<h1>Hello ' + request.GET['name'] + '</h1>'\n",
         "XSS via unescaped query param"),
        ("command_injection", "runner.py", 4,
         "def run(cmd):\n     import subprocess\n     subprocess.call('git ' + cmd, shell=True)\n",
         "Command injection via shell=True"),
        ("path_traversal", "server.py", 3,
         "def serve(filename):\n     return open('static/' + filename).read()\n",
         "Path traversal to read arbitrary files"),
        ("insecure_deserialization", "load.py", 3,
         "def load(data):\n     import pickle\n     return pickle.loads(data)\n",
         "Insecure pickle deserialization"),
        ("weak_crypto", "auth.py", 2,
         "def hash_pw(pw):\n     import hashlib\n     return hashlib.md5(pw.encode()).hexdigest()\n",
         "MD5 for password hashing"),
        ("hardcoded_secret", "config.py", 1,
         "API_KEY = 'sk-live-12345abcdef67890'\n",
         "Hardcoded API key"),
        ("ssrf", "fetch.py", 3,
         "def fetch(url):\n     import urllib\n     return urllib.urlopen(url).read()\n",
         "SSRF via user-controlled URL"),
        ("resource_leak", "file_handler.py", 2,
         "def read_all(path):\n     f = open(path)\n     return f.read()\n",
         "File handle not closed"),
        ("insecure_eval", "calc.py", 2,
         "def calc(expr):\n     return eval(expr)\n",
         "Code injection via eval"),
    ]
    cases = []
    for vtype, fname, line, code, desc in vulns:
        # 10 cases per type
        for i in range(10):
            cases.append({
                "id": f"sec_vuln_{vtype}_{i+1:02d}",
                "group": f"eval/sec/vuln/{vtype}/{i}",
                "prompt": f"Review this diff:\n\n@@ -1,2 +1,3 @@\n+{code}\n",
                "expected_status": "REJECT",
                "expected_type": vtype,
                "expected_line": line,
                "description": desc,
            })
    return cases

# ── SECURITY CLEAN suite ───────────────────────────────────────
def gen_security_clean():
    cleans = [
        ("safe_query", "user.py", "def get_user(db, uid):\n     return db.execute('SELECT * FROM users WHERE id=?', (uid,)).fetchone()\n"),
        ("safe_hash", "auth.py", "def hash_pw(pw):\n     import bcrypt\n     return bcrypt.hashpw(pw.encode(), bcrypt.gensalt()).decode()\n"),
        ("safe_file", "handler.py", "def read_file(path):\n     with open(path) as f:\n         return f.read()\n"),
        ("safe_render", "page.py", "def render(name):\n     from html import escape\n     return '<h1>' + escape(name) + '</h1>'\n"),
        ("safe_exec", "runner.py", "def run(cmd):\n     import subprocess\n     subprocess.run(cmd, shell=False)\n"),
        ("safe_load", "config.py", "def load(path):\n     import json\n     with open(path) as f:\n         return json.load(f)\n"),
        ("safe_crypto", "token.py", "def sign(data):\n     import hmac, hashlib\n     return hmac.new(key, data, hashlib.sha256).hexdigest()\n"),
        ("safe_http", "api.py", "def get(url):\n     from urllib.parse import urlparse\n     assert urlparse(url).scheme in ('http', 'https')\n     return requests.get(url).text\n"),
        ("safe_env", "secret.py", "def get_key():\n     import os\n     return os.environ['API_KEY']\n"),
        ("safe_regex", "validate.py", "def validate(email):\n     import re\n     return bool(re.match(r'^[\\w.+-]+@[\\w-]+\\.[a-z]{2,}$', email))\n"),
    ]
    cases = []
    for ctype, fname, code in cleans:
        for i in range(10):
            cases.append({
                "id": f"sec_clean_{ctype}_{i+1:02d}",
                "group": f"eval/sec/clean/{ctype}/{i}",
                "prompt": f"Review this diff:\n\n@@ -1,2 +1,4 @@\n+{code}\n",
                "expected_status": "PASS",
                "description": f"Safe code of type {ctype}",
            })
    return cases

# ── PLAN_JSON suite ────────────────────────────────────────────
def gen_plan_json():
    cases = []
    for i in range(50):
        cases.append({
            "id": f"plan_json_{i+1:03d}",
            "group": f"eval/plan/batch1/task{i}",
            "prompt": f"Plan the implementation for feature #{i}: add user authentication with OAuth2. Return as JSON with status, target_version, critique, final_plan fields.",
            "expected_status_values": ["APPROVED", "NEEDS_REVISION"],
            "required_fields": ["status", "target_version", "critique", "final_plan"],
        })
    return cases

if __name__ == "__main__":
    write_suite("tool_syntax", "v001", gen_tool_syntax(),
                {"description": "Single and multi-tool call parsing"})
    write_suite("anti_loop", "v001", gen_anti_loop(),
                {"description": "Error injection and loop detection"})
    write_suite("security_vuln", "v001", gen_security_vuln(),
                {"description": "Vulnerability detection", "min_per_type": 5})
    write_suite("security_clean", "v001", gen_security_clean(),
                {"description": "Clean code should not be REJECTed"})
    write_suite("plan_json", "v001", gen_plan_json(),
                {"description": "Plan JSON structure and status"})
    print("\nAll suites generated.")
