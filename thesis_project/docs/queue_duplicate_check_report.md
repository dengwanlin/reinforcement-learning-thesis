Project: Reinforcement Learning Experiment Queue System
Component: queue.db (SQLite job queue)
Author: Wang
Date: 2025-10-28

1️⃣ Background

During large-scale experiment scheduling, it is common that identical commands are imported multiple times into the SQLite job queue (queue.db).
Each record corresponds to a command line stored in the column cmd within the table jobs.

Duplicate commands may cause:

Re-execution of identical experiments,

Inconsistent status statistics (Done count vs. actual runs),

Redundant disk usage and confusing results.

This document describes how to detect, inspect, clean, and prevent duplicate command entries.

2️⃣ Quick Overview of the jobs Table
Column	Type	Description
id	INTEGER PRIMARY KEY	Unique ID per job
cmd	TEXT	Full training command line
status	INTEGER	0 = queued, 1 = done, 2 = running
attempts	INTEGER	Number of attempts
rc	INTEGER	Return code of last execution
stdout_path / stderr_path	TEXT	Log file paths
host, pid	TEXT / INTEGER	Execution metadata
3️⃣ Detecting Duplicates
a) Check for duplicate entries count
sqlite3 queue.db "
SELECT COUNT(*) AS total,
       COUNT(DISTINCT cmd) AS distinct_count,
       COUNT(*) - COUNT(DISTINCT cmd) AS duplicates
FROM jobs;
"


Example output:

total        distinct_count   duplicates
12240        12200            40


→ 40 duplicate command lines detected.

b) List duplicate commands and their frequencies
sqlite3 queue.db "
SELECT cmd, COUNT(*) AS n
FROM jobs
GROUP BY cmd
HAVING n > 1
ORDER BY n DESC
LIMIT 20;
"


Example:

python runner.py --algo a2c --env CartPole-v1 ... | 3
python runner.py --algo ppo --env LunarLander-v3 ... | 2

c) View duplicates with their status distribution
sqlite3 queue.db "
SELECT cmd,
       GROUP_CONCAT(status) AS statuses,
       COUNT(*) AS n
FROM jobs
GROUP BY cmd
HAVING n > 1
ORDER BY n DESC
LIMIT 20;
"


Example:

python runner.py --algo a2c ... | 1,0 | 2


→ one job finished (1), one still queued (0).

4️⃣ Removing Duplicate Entries

To keep only the earliest record of each command and remove all duplicates:

sqlite3 queue.db "
DELETE FROM jobs
WHERE id NOT IN (
  SELECT MIN(id)
  FROM jobs
  GROUP BY cmd
);
VACUUM;
"


✅ Effect:
Each command (cmd) remains only once in the database.
This is safe and recommended after large batch imports.

5️⃣ Optional: Marking Duplicates Instead of Deleting

If you prefer to flag duplicates for later review:

sqlite3 queue.db "
ALTER TABLE jobs ADD COLUMN duplicate INTEGER DEFAULT 0;
UPDATE jobs
SET duplicate = 1
WHERE id NOT IN (SELECT MIN(id) FROM jobs GROUP BY cmd);
"


Then you can check:

sqlite3 queue.db "SELECT COUNT(*) FROM jobs WHERE duplicate=1;"

6️⃣ Command-Line Alternative (without SQL shell)
a) Count number of duplicate commands
sqlite3 -noheader -separator '|' queue.db "SELECT cmd FROM jobs" | sort | uniq -d | wc -l

b) Export duplicates to a text file
sqlite3 -noheader -separator '|' queue.db "SELECT cmd FROM jobs" | sort | uniq -d > duplicate_cmds.txt
head duplicate_cmds.txt

7️⃣ Permanent Prevention of Future Duplicates

Add a unique index to enforce command-line uniqueness in queue.db:

CREATE UNIQUE INDEX IF NOT EXISTS idx_unique_cmd ON jobs(cmd);


Then, in your import script (init_from_txt.py), use:

conn.execute(
    "INSERT OR IGNORE INTO jobs(cmd, status) VALUES(?, ?)", 
    (cmd, status)
)


✅ Effect:

New unique commands are inserted normally.

Duplicates are silently ignored during import.

No manual cleanup is needed afterward.

8️⃣ Exporting a Duplicate Report (for audit)

Generate a CSV listing all duplicates and their details:

sqlite3 -header -csv queue.db "
SELECT id, status, attempts, rc, cmd
FROM jobs
WHERE cmd IN (
  SELECT cmd FROM jobs GROUP BY cmd HAVING COUNT(*) > 1
)
ORDER BY cmd, id;
" > duplicate_jobs_report.csv


You can then open duplicate_jobs_report.csv in Excel or any data viewer.

9️⃣ Verification Steps

After cleanup:

sqlite3 queue.db "
SELECT COUNT(*) AS total,
       COUNT(DISTINCT cmd) AS distinct_count,
       COUNT(*) - COUNT(DISTINCT cmd) AS duplicates
FROM jobs;
"


Expected result:

duplicates = 0


You can also re-import the same command file safely.
If duplicates were present, the database will simply skip them due to the unique index.

🔟 Summary Table
Goal	Command
Count total duplicates	SELECT COUNT(*) - COUNT(DISTINCT cmd) FROM jobs;
List duplicate commands	SELECT cmd, COUNT(*) FROM jobs GROUP BY cmd HAVING COUNT(*)>1;
Delete duplicates, keep first	DELETE FROM jobs WHERE id NOT IN (SELECT MIN(id) FROM jobs GROUP BY cmd);
Mark duplicates	Add column duplicate, update where duplicates found
Prevent future duplicates	Add unique index idx_unique_cmd + INSERT OR IGNORE
Export duplicate report	duplicate_jobs_report.csv
✅ Conclusion

The queue system stores all job commands in a single table (jobs), which can accumulate duplicates over multiple imports.

Using simple SQL queries, one can identify and clean duplicates safely.

The best long-term solution is to enforce a unique index and use INSERT OR IGNORE in import scripts.

Regular integrity checks (COUNT(*) - COUNT(DISTINCT cmd)) ensure that the database remains clean, consistent, and reproducible for all reinforcement-learning experiment batches.
