import json
import sqlite3
from backend.app.db import init_db, connect
from backend.app.services import io_jsonl

init_db('data/examples.db')
conn = connect('data/examples.db')

for cat in ['tool', 'loop', 'plan', 'sec']:
    with open(f'data/gold/{cat}.jsonl', 'r') as f:
        text = f.read()
    result = io_jsonl.import_lines(conn, text, default_status='approved')
    print(f'{cat}: imported={result["imported"]}, failed={result["failed"]}')
    if result["errors"]:
        for e in result["errors"][:5]:
            print(f'  line {e["line"]}: {e["error"]}')

conn.close()