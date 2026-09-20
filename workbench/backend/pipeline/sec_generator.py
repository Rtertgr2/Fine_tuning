"""Security-category generator (T2.7): vulnerability review examples.

Generates diffs with known vulnerabilities (from CWE patterns) and clean diffs,
with ground-truth labels from the injection (not model guesses).
"""

from __future__ import annotations

import json
import random
from typing import Any, Sequence

from backend.pipeline.generator import BaseGenerator

SYSTEM_PROMPT = """You are a security reviewer. Review the following diff and respond with JSON only:
{"status": "PASS" or "REJECT", "issues": [{"file": "...", "line": N, "type": "...", "severity": "...", "fix": "..."}]}"""

# Vulnerability patterns for synthetic generation
VULN_PATTERNS = {
    "sql_injection": {
        "cwe": "CWE-89",
        "diff": (
            "diff --git a/app.py b/app.py\n"
            "--- a/app.py\n"
            "+++ b/app.py\n"
            "@@ -1,5 +1,6 @@\n"
            " import sqlite3\n"
            " def get_user(name):\n"
            "-    query = 'SELECT * FROM users WHERE name = ?'\n"
            "-    cur.execute(query, (name,))\n"
            "+    query = \"SELECT * FROM users WHERE name = '\" + name + \"'\"\n"
            "+    cur.execute(query)\n"
            "     return cur.fetchone()\n"
        ),
        "issues": [
            {
                "file": "app.py",
                "line": 3,
                "type": "sql_injection",
                "severity": "critical",
                "fix": "ใช้ parameterized query: cur.execute('SELECT * FROM users WHERE name = ?', (name,))",
            }
        ],
    },
    "hardcoded_secret": {
        "cwe": "CWE-798",
        "diff": (
            "diff --git a/config.py b/config.py\n"
            "--- a/config.py\n"
            "+++ b/config.py\n"
            "@@ -1,3 +1,4 @@\n"
            " import os\n"
            "-API_KEY = os.environ.get('API_KEY')\n"
            "+API_KEY = 'sk-proj-abc123def456ghi789'\n"
            " DEBUG = False\n"
        ),
        "issues": [
            {
                "file": "config.py",
                "line": 2,
                "type": "hardcoded_secret",
                "severity": "high",
                "fix": "ใช้ os.environ.get('API_KEY') แทน hardcoded value",
            }
        ],
    },
    "path_traversal": {
        "cwe": "CWE-22",
        "diff": (
            "diff --git a/server.py b/server.py\n"
            "--- a/server.py\n"
            "+++ b/server.py\n"
            "@@ -10,5 +10,5 @@\n"
            " @app.route('/files/<path>')\n"
            " def serve_file(path):\n"
            '-    safe = os.path.join(BASE_DIR, os.path.normpath(path))\n'
            "-    return open(safe, 'rb').read()\n"
            "+    return open('/var/www/' + path, 'rb').read()\n"
        ),
        "issues": [
            {
                "file": "server.py",
                "line": 12,
                "type": "path_traversal",
                "severity": "high",
                "fix": "ใช้ os.path.join(BASE_DIR, os.path.normpath(path)) และตรวจสอบว่าอยู่ใน BASE_DIR",
            }
        ],
    },
    "xss": {
        "cwe": "CWE-79",
        "diff": (
            "diff --git a/templates/page.html b/templates/page.html\n"
            "--- a/templates/page.html\n"
            "+++ b/templates/page.html\n"
            "@@ -5,3 +5,3 @@\n"
            "-<div>{{ user_input | safe }}</div>\n"
            "+<div>{{ user_input }}</div>\n"
        ),
        "issues": [
            {
                "file": "templates/page.html",
                "line": 5,
                "type": "xss",
                "severity": "high",
                "fix": "ลบ | safe filter เพื่อให้ Jinja2 auto-escape HTML",
            }
        ],
    },
    "insecure_deserialization": {
        "cwe": "CWE-502",
        "diff": (
            "diff --git a/handler.py b/handler.py\n"
            "--- a/handler.py\n"
            "+++ b/handler.py\n"
            "@@ -3,3 +3,3 @@\n"
            "-import json\n"
            "-data = json.loads(payload)\n"
            "+import pickle\n"
            "+data = pickle.loads(payload)\n"
        ),
        "issues": [
            {
                "file": "handler.py",
                "line": 4,
                "type": "insecure_deserialization",
                "severity": "critical",
                "fix": "ใช้ json.loads แทน pickle.loads เพื่อป้องกัน arbitrary code execution",
            }
        ],
    },
    "resource_leak": {
        "cwe": "CWE-401",
        "diff": (
            "diff --git a/db.py b/db.py\n"
            "--- a/db.py\n"
            "+++ b/db.py\n"
            "@@ -1,5 +1,4 @@\n"
            " def query(sql):\n"
            "-    conn = get_connection()\n"
            "-    return conn.execute(sql).fetchall()\n"
            "+    return get_connection().execute(sql).fetchall()\n"
        ),
        "issues": [
            {
                "file": "db.py",
                "line": 2,
                "type": "resource_leak",
                "severity": "medium",
                "fix": "ใช้ context manager (with statement) เพื่อปิด connection อัตโนมัติ",
            }
        ],
    },
}

# Clean diffs (no vulnerabilities)
CLEAN_DIFFS = [
    {
        "diff": (
            "diff --git a/utils.py b/utils.py\n"
            "--- a/utils.py\n"
            "+++ b/utils.py\n"
            "@@ -10,3 +10,6 @@\n"
            " def format_name(first, last):\n"
            "-    return first + ' ' + last\n"
            "+    if not first or not last:\n"
            "+        return first or last\n"
            "+    return f'{first} {last}'\n"
        ),
    },
    {
        "diff": (
            "diff --git a/test_math.py b/test_math.py\n"
            "--- a/test_math.py\n"
            "+++ b/test_math.py\n"
            "@@ -1,3 +1,6 @@\n"
            " def test_add():\n"
            "-    assert add(1, 2) == 3\n"
            "+    assert add(1, 2) == 3\n"
            "+    assert add(-1, 1) == 0\n"
            "+    assert add(0, 0) == 0\n"
        ),
    },
    {
        "diff": (
            "diff --git a/README.md b/README.md\n"
            "--- a/README.md\n"
            "+++ b/README.md\n"
            "@@ -1,3 +1,5 @@\n"
            " # My Project\n"
            "\n"
            "+## Installation\n"
            "+pip install -r requirements.txt\n"
            "\n"
            " ## Usage\n"
        ),
    },
]


class SecGenerator(BaseGenerator):
    """Generate security-category examples with known vulnerability labels."""

    category = "sec"
    source = "generated"

    def __init__(self, db_path=None, budget_usd=None, vuln_types=None, pass_ratio=0.5):
        super().__init__(db_path, budget_usd)
        self._vuln_types = vuln_types or list(VULN_PATTERNS.keys())
        self._pass_ratio = pass_ratio  # target fraction of PASS examples

    def _generate_candidates(
        self,
        seeds: Sequence[dict[str, Any]],
        trajectory_factory: Any | None = None,
    ) -> list[dict[str, Any]]:
        candidates = []
        for seed in seeds:
            # Decide if this should be a PASS or REJECT
            is_pass = random.random() < self._pass_ratio
            vuln_type = "unknown"

            if is_pass:
                traj = self._generate_clean_review(seed)
            else:
                vuln_type = random.choice(self._vuln_types)
                traj = self._generate_vuln_review(seed, vuln_type)

            if traj is not None:
                candidates.append({
                    "category": "sec",
                    "messages": traj["messages"],
                    "tools": None,
                    "group_id": seed.get("group_id", f"gen/sec/{'clean' if is_pass else vuln_type}"),
                    "meta": {
                        "vulnerability_type": None if is_pass else vuln_type,
                        "status": traj["status"],
                        "cwe": traj.get("cwe"),
                    },
                })
        return candidates

    def _generate_clean_review(self, seed: dict[str, Any]) -> dict[str, Any] | None:
        """Generate a PASS review for a clean diff."""
        clean = random.choice(CLEAN_DIFFS)
        diff = seed.get("diff", clean["diff"])

        response = {"status": "PASS", "issues": []}

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"ตรวจ diff นี้:\n{diff}"},
            {"role": "assistant", "content": json.dumps(response, ensure_ascii=False)},
        ]

        return {"messages": messages, "status": "PASS"}

    def _generate_vuln_review(self, seed: dict[str, Any], vuln_type: str) -> dict[str, Any] | None:
        """Generate a REJECT review with known vulnerability."""
        pattern = VULN_PATTERNS[vuln_type]
        diff = seed.get("diff", pattern["diff"])
        issues = pattern["issues"]

        response = {"status": "REJECT", "issues": issues}

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"ตรวจ diff นี้:\n{diff}"},
            {"role": "assistant", "content": json.dumps(response, ensure_ascii=False)},
        ]

        return {
            "messages": messages,
            "status": "REJECT",
            "cwe": pattern["cwe"],
        }
