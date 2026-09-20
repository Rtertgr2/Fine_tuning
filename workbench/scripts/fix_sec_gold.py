#!/usr/bin/env python3
"""Generate additional unique sec examples to:
- Reach >=15 unique sec examples
- Get PASS share within 35-65%
- Cover more vulnerability types
Then deduplicate the full sec.jsonl.
"""
import json
from pathlib import Path

GOLD = Path("data/gold")

SEC_SYSTEM = "You are a security review agent. Analyze the provided code diff for security issues. Return your analysis as JSON with fields: status (PASS or REJECT), and if REJECT, issues[] with file, line, type, severity, fix."

def sec_ex(exid, user_diff, assistant_json, group):
    return {
        "id": exid,
        "category": "sec",
        "messages": [
            {"role": "system", "content": SEC_SYSTEM},
            {"role": "user", "content": user_diff},
            {"role": "assistant", "content": assistant_json},
        ],
        "tools": [],
        "meta": {"source": "manual", "group": group, "status": "approved"},
    }

# Read existing
existing = []
with open(GOLD / "sec.jsonl") as f:
    for line in f:
        existing.append(json.loads(line))

# New unique examples to add
new_examples = [
    # PASS: safe parameterized query
    sec_ex("ex_gold_sec_new_001",
        "Review this diff:\n\n@@ -1,3 +1,4 @@\n+import sqlite3\n def get_user(conn, uid):\n-    return conn.execute('SELECT * FROM users WHERE id=' + str(uid)).fetchone()\n+    return conn.execute('SELECT * FROM users WHERE id=?', (uid,)).fetchone()\n",
        json.dumps({"status": "PASS"}),
        "gold/sec/pass-param-1"),
    
    # PASS: safe password hashing
    sec_ex("ex_gold_sec_new_002",
        "Review this diff:\n\n@@ -1,3 +1,5 @@\n+import bcrypt\n def hash_password(pwd):\n-    return pwd\n+    return bcrypt.hashpw(pwd.encode(), bcrypt.gensalt())\n",
        json.dumps({"status": "PASS"}),
        "gold/sec/pass-hash-1"),
    
    # PASS: safe input validation
    sec_ex("ex_gold_sec_new_003",
        "Review this diff:\n\n@@ -1,3 +1,5 @@\n+import re\n def validate_email(email):\n-    return True\n+    return bool(re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$', email))\n",
        json.dumps({"status": "PASS"}),
        "gold/sec/pass-regex-1"),
    
    # PASS: safe file handling with context manager
    sec_ex("ex_gold_sec_new_004",
        "Review this diff:\n\n@@ -1,3 +1,4 @@\n def read_config(path):\n-    f = open(path)\n-    return f.read()\n+    with open(path) as f:\n+        return f.read()\n",
        json.dumps({"status": "PASS"}),
        "gold/sec/pass-ctx-mgr-1"),
    
    # PASS: safe secret management
    sec_ex("ex_gold_sec_new_005",
        "Review this diff:\n\n@@ -1,2 +1,3 @@\n+import os\n-SECRET_KEY = 'hardcoded-secret'\n+SECRET_KEY = os.environ.get('SECRET_KEY', 'changeme')\n",
        json.dumps({"status": "PASS"}),
        "gold/sec/pass-env-secret-1"),
    
    # REJECT: SSRF vulnerability
    sec_ex("ex_gold_sec_new_006",
        "Review this diff:\n\n@@ -1,3 +1,5 @@\n+import requests\n def fetch_url(url):\n-    return None\n+    return requests.get(url).text\n",
        json.dumps({"status": "REJECT", "issues": [
            {"file": "app.py", "line": 4, "type": "owasp_a10_ssrf", "severity": "high",
             "fix": "Validate and whitelist allowed URL schemes and domains before fetching"}
        ]}),
        "gold/sec/reject-ssrf-1"),
    
    # REJECT: insecure deserialization (different diff)
    sec_ex("ex_gold_sec_new_007",
        "Review this diff:\n\n@@ -1,3 +1,4 @@\n+import yaml\n def load_config(path):\n-    return {}\n+    return yaml.load(open(path))\n",
        json.dumps({"status": "REJECT", "issues": [
            {"file": "config.py", "line": 4, "type": "insecure_deserialization", "severity": "critical",
             "fix": "Use yaml.safe_load() instead of yaml.load() to prevent arbitrary code execution"}
        ]}),
        "gold/sec/reject-yaml-1"),
    
    # REJECT: SQL injection (different context)
    sec_ex("ex_gold_sec_new_008",
        "Review this diff:\n\n@@ -1,3 +1,4 @@\n def search_products(db, query):\n-    return []\n+    return db.execute(f\"SELECT * FROM products WHERE name LIKE '%{query}%'\").fetchall()\n",
        json.dumps({"status": "REJECT", "issues": [
            {"file": "products.py", "line": 3, "type": "sql_injection", "severity": "critical",
             "fix": "Use parameterized query: db.execute('SELECT * FROM products WHERE name LIKE ?', ('%' + query + '%',))"}
        ]}),
        "gold/sec/reject-sql-2"),
    
    # REJECT: broken access control
    sec_ex("ex_gold_sec_new_009",
        "Review this diff:\n\n@@ -1,3 +1,4 @@\n+from flask import request\n def delete_user(user_id):\n-    pass\n+    db.delete(user_id)\n",
        json.dumps({"status": "REJECT", "issues": [
            {"file": "routes.py", "line": 4, "type": "owasp_a01_broken_access_control", "severity": "critical",
             "fix": "Verify that the requesting user has permission to delete the target user before executing the delete"}
        ]}),
        "gold/sec/reject-access-2"),
    
    # REJECT: cryptographic failure (weak hash)
    sec_ex("ex_gold_sec_new_010",
        "Review this diff:\n\n@@ -1,3 +1,3 @@\n def verify_token(token):\n-    return hashlib.md5(token.encode()).hexdigest()\n+    return hashlib.sha1(token.encode()).hexdigest()\n",
        json.dumps({"status": "REJECT", "issues": [
            {"file": "auth.py", "line": 3, "type": "owasp_a02_crypto_failures", "severity": "high",
             "fix": "Use a cryptographically secure comparison with hmac.compare_digest and SHA-256 or better"}
        ]}),
        "gold/sec/reject-crypto-2"),
]

# Add new examples
for ex in new_examples:
    existing.append(ex)

# Deduplicate by content hash
import hashlib
seen_hashes = set()
unique = []
dups = 0
for ex in existing:
    msgs = ex["messages"]
    tools = ex.get("tools")
    norm = json.dumps(msgs, sort_keys=True, ensure_ascii=False)
    if tools:
        norm += json.dumps(tools, sort_keys=True, ensure_ascii=False)
    h = hashlib.sha256(norm.encode()).hexdigest()
    if h not in seen_hashes:
        seen_hashes.add(h)
        unique.append(ex)
    else:
        dups += 1

# Write back
with open(GOLD / "sec.jsonl", "w") as f:
    for ex in unique:
        f.write(json.dumps(ex, ensure_ascii=False) + "\n")

# Count PASS/REJECT
pass_count = sum(1 for ex in unique if any(
    m["role"] == "assistant" and '"PASS"' in str(m.get("content", ""))
    for m in ex["messages"]
))
reject_count = len(unique) - pass_count

print("sec.jsonl: %d unique (removed %d duplicates)" % (len(unique), dups))
print("  PASS: %d (%.0f%%)" % (pass_count, pass_count/len(unique)*100))
print("  REJECT: %d (%.0f%%)" % (reject_count, reject_count/len(unique)*100))
