#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, sqlite3, time, os

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  cmd TEXT NOT NULL,
  status INTEGER NOT NULL DEFAULT 0,   -- 0=queued, 1=done, 2=running
  attempts INTEGER NOT NULL DEFAULT 0,
  rc INTEGER,
  last_start REAL,
  last_end REAL,
  stdout_path TEXT,
  stderr_path TEXT,
  host TEXT,
  pid INTEGER
);
CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
"""

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cmd-file", required=True, help="text commands file, one per line")
    p.add_argument("--db", required=True, help="the path of SQLite, such as queue.db")
    args = p.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.db)), exist_ok=True)
    conn = sqlite3.connect(args.db)
    conn.executescript(SCHEMA)

    with open(args.cmd_file, "r", encoding="utf-8") as f, conn:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # allow status prefix, e.g. "1|python train.py ..."
            status = 0
            cmd = line
            if "|" in line[:3]:
                maybe, rest = line.split("|", 1)
                if maybe.strip() in ("0","1"):
                    status = int(maybe.strip())
                    cmd = rest.strip()
            conn.execute("INSERT OR IGNORE INTO jobs(cmd,status) VALUES(?,?)", (cmd, status))

    conn.commit()
    cur = conn.execute("SELECT COUNT(*), SUM(CASE WHEN status=1 THEN 1 ELSE 0 END) FROM jobs")
    total, done = cur.fetchone()
    print(f"Imported {total} commands (done={done or 0}). DB -> {args.db}")

if __name__ == "__main__":
    main()
