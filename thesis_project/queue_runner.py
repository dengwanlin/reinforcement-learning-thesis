#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, os, sqlite3, time, subprocess, shlex, socket, sys, traceback, signal
from datetime import datetime

# ------------------------------- Utils & DB -------------------------------

def now_ts() -> float:
    return time.time()

def with_conn(db: str):
    conn = sqlite3.connect(db, timeout=60.0, isolation_level=None)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn

# ----------------------------- Claim / Finish -----------------------------

def claim_job(conn: sqlite3.Connection, max_retries: int | None = None):
    """
    Preempt a task with status=0 -> set it to 2 and return (id, cmd, attempts).
Tasks whose attempts have reached the upper limit will be skipped; tasks will be claimed in order of id.
    """
    for _ in range(5):
        try:
            conn.execute("BEGIN IMMEDIATE;")
            if max_retries is not None and max_retries >= 0:
                cur = conn.execute("""
                    SELECT id, cmd, attempts
                    FROM jobs
                    WHERE status=0 AND attempts < ?
                    ORDER BY id
                    LIMIT 1
                """, (max_retries,))
            else:
                cur = conn.execute("""
                    SELECT id, cmd, attempts
                    FROM jobs
                    WHERE status=0
                    ORDER BY id
                    LIMIT 1
                """)
            row = cur.fetchone()
            if not row:
                conn.execute("COMMIT;")
                return None
            job_id, cmd, attempts = row
            conn.execute(
                "UPDATE jobs SET status=2, last_start=? WHERE id=?",
                (now_ts(), job_id)
            )
            conn.execute("COMMIT;")
            return job_id, cmd, attempts
        except sqlite3.OperationalError:
            conn.execute("ROLLBACK;")
            time.sleep(0.05)
    return None

def finish_job(conn: sqlite3.Connection, job_id: int, rc: int, stdout_path: str, stderr_path: str, running_pid: int, ok_rcs: set[int] = {0}):
    status_val = 1 if rc in ok_rcs else 0
    conn.execute(
        "UPDATE jobs SET status=?, rc=?, last_end=?, stdout_path=?, stderr_path=?, pid=? WHERE id=?",
        (status_val, rc, now_ts(), stdout_path, stderr_path, running_pid, job_id)
    )

# ----------------------------- Process Spawn ------------------------------

def spawn_process(cmd: str, workdir: str, env: dict, stdout_fh, stderr_fh, use_shell: bool, strict_bash: bool):
    """
When use_shell=False , shlex.split is used; when use_shell=True , it is allowed to be executed as is.
When strict_bash=True , it is wrapped with `bash -lc 'set -euo pipefail; CMD'` to improve rc reliability.
    """
    if use_shell:
        final_cmd = cmd
        if strict_bash:
            final_cmd = f"bash -lc 'set -euo pipefail; {cmd}'"
        return subprocess.Popen(
            final_cmd,
            cwd=workdir,
            shell=True,
            env=env,
            stdout=stdout_fh,
            stderr=stderr_fh,
        )
    else:
        argv = shlex.split(cmd)
        return subprocess.Popen(
            argv,
            cwd=workdir,
            shell=False,
            env=env,
            stdout=stdout_fh,
            stderr=stderr_fh,
        )

# ------------------------------ Worker Loop -------------------------------

def worker_loop(db, workdir, env_extra, max_retries, logdir, use_shell: bool, strict_bash: bool, ok_rcs: set[int], timeout: float | None):
    host = socket.gethostname()
    conn = with_conn(db)
    os.makedirs(logdir, exist_ok=True)

    current_job_id: int | None = None

    # successfully return with SIGTERM/SIGINT
    stop_flag = {"stop": False}
    def _sig_handler(signum, frame):
        stop_flag["stop"] = True
    signal.signal(signal.SIGTERM, _sig_handler)
    signal.signal(signal.SIGINT, _sig_handler)

    try:
        while not stop_flag["stop"]:
            item = claim_job(conn, max_retries)
            if not item:
                return
            job_id, cmd, attempts = item
            current_job_id = job_id

            # the log paths
            tag = f"job_{job_id:06d}"
            stdout_path = os.path.join(logdir, f"{tag}.out")
            stderr_path = os.path.join(logdir, f"{tag}.err")

            # update attempts, host, pid (pid will be updated again after spawn)
            conn.execute(
                "UPDATE jobs SET attempts=attempts+1, host=?, pid=? WHERE id=?",
                (host, os.getpid(), job_id)
            )

            env = os.environ.copy()
            env.update(env_extra)

            proc = None
            try:
                with open(stdout_path, "ab") as so, open(stderr_path, "ab") as se:
                    proc = spawn_process(cmd, workdir, env, so, se, use_shell=use_shell, strict_bash=strict_bash)
                    # change pid after spawn
                    conn.execute("UPDATE jobs SET pid=? WHERE id=?", (proc.pid, job_id))
                    if timeout is not None and timeout > 0:
                        try:
                            rc = proc.wait(timeout=timeout)
                        except subprocess.TimeoutExpired:
                            # out of time: kill it
                            proc.kill()
                            rc = 124  # we use 124 as timeout rc
                    else:
                        rc = proc.wait()
                finish_job(conn, job_id, rc, stdout_path, stderr_path, proc.pid, ok_rcs=ok_rcs)
            except Exception as e:
                # when exception occurs, log it to stderr_path
                try:
                    with open(stderr_path, "ab") as se:
                        se.write(b"\n[queue_runner exception]\n")
                        se.write("".join(traceback.format_exception(e)).encode("utf-8", errors="ignore"))
                except Exception:
                    pass
                finish_job(conn, job_id, rc=9999, stdout_path=stdout_path, stderr_path=stderr_path, running_pid=(proc.pid if proc else os.getpid()), ok_rcs=ok_rcs)
            finally:
                current_job_id = None

            if stop_flag["stop"]:
                break

    finally:
        # if we are exiting while having a claimed job, reset it to queued
        if current_job_id is not None:
            conn.execute("UPDATE jobs SET status=0, pid=NULL, host=NULL WHERE id=?", (current_job_id,))

# -------------------------------- Commands --------------------------------

def cmd_run(args):
    # enverionment variables to set for each worker
    env_extra = {
        "CUDA_VISIBLE_DEVICES": "",   # disable GPU usage
        "TF_CPP_MIN_LOG_LEVEL": "3",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
    }

    logdir = args.logdir or os.path.join(os.path.dirname(os.path.abspath(args.db)), "logs")
    os.makedirs(logdir, exist_ok=True)

    ok_rcs = {int(x) for x in args.ok_rcs.split(",")} if args.ok_rcs else {0}

    from multiprocessing import Process
    workers = []
    for _ in range(args.workers):
        p = Process(
            target=worker_loop,
            args=(args.db, args.workdir, env_extra, args.max_retries, logdir, args.use_shell, args.strict_bash, ok_rcs, args.timeout),
        )
        p.start()
        workers.append(p)

    for p in workers:
        p.join()

    print("All workers exited.")

def cmd_status(args):
    conn = with_conn(args.db)
    cur = conn.execute("SELECT COUNT(*) FROM jobs"); total = cur.fetchone()[0]
    cur = conn.execute("SELECT COUNT(*) FROM jobs WHERE status=1"); done = cur.fetchone()[0]
    cur = conn.execute("SELECT COUNT(*) FROM jobs WHERE status=0"); queued = cur.fetchone()[0]
    cur = conn.execute("SELECT COUNT(*) FROM jobs WHERE status=2"); running = cur.fetchone()[0]
    print(f"Total: {total} | Done(1): {done} | Queued(0): {queued} | Running(2): {running}")
    # attempts histogram
    hist = dict(conn.execute("SELECT attempts, COUNT(*) FROM jobs GROUP BY attempts ORDER BY attempts").fetchall())
    print("Attempts histogram:", hist)

def cmd_export(args):
    conn = with_conn(args.db)
    sql = "SELECT status, cmd FROM jobs ORDER BY id"
    params = ()
    if args.only_status is not None:
        sql = "SELECT status, cmd FROM jobs WHERE status=? ORDER BY id"
        params = (args.only_status,)
    with open(args.out, "w", encoding="utf-8") as f:
        for status, cmd in conn.execute(sql, params):
            f.write(f"{status}|{cmd}\n")
    print(f"Exported -> {args.out}")

def cmd_failed(args):
    conn = with_conn(args.db)
    rows = conn.execute("""
        SELECT id, rc, attempts, cmd
        FROM jobs
        WHERE status=0 AND attempts>0
        ORDER BY id
    """).fetchall()
    if not rows:
        print("No failed jobs.")
        return
    for r in rows:
        print(f"[id={r[0]}] rc={r[1]} attempts={r[2]} :: {r[3]}")

# ------------------------------ Recover Stuck ------------------------------

def pid_alive_on_this_host(pid: int) -> bool:
    if pid is None or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        # if we don't have permission to signal the process, assume it's alive
        return True

def cmd_recover(args):
    now = time.time()
    stale_sec = args.stale_seconds
    this_host = socket.gethostname()
    conn = with_conn(args.db)

    rows = conn.execute("""
        SELECT id, host, pid, last_start FROM jobs WHERE status=2
    """).fetchall()

    recovered = 0
    for job_id, host, pid, last_start in rows:
        should_recover = False
        # for different hosts, pid does not matter
        if host and host != this_host:
            should_recover = True
        # if on this host, check if pid is alive
        elif pid and not pid_alive_on_this_host(pid):
            should_recover = True
        # over stale time
        elif last_start and (now - float(last_start) > stale_sec):
            should_recover = True

        if should_recover:
            conn.execute("""
                UPDATE jobs SET status=0, pid=NULL, host=NULL WHERE id=?
            """, (job_id,))
            recovered += 1

    print(f"Recovered {recovered} stuck running jobs back to queue.")

# --------------------------------- Argparse --------------------------------

def main():
    ap = argparse.ArgumentParser(description="SQLite concurrent task queue (0/1/2 status)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("run", help="Start concurrent execution")
    sp.add_argument("--db", required=True)
    sp.add_argument("--workers", type=int, default=32)
    sp.add_argument("--workdir", required=True, help="The working directory where runner.py is located")
    sp.add_argument("--max-retries", type=int, default=2, help="The maximum number of retries upon failure; -1 means infinite retries")
    sp.add_argument("--logdir", help="Log directory; default is <db peer>/logs")
    sp.add_argument("--use-shell", action="store_true", help="Execute commands using the shell (allows pipe/redirection syntax, etc.)")
    sp.add_argument("--strict-bash", action="store_true", help="With --use-shell: use bash -lc 'set -euo pipefail; CMD' strict mode wrapper")
    sp.add_argument("--ok-rcs", type=str, default="", help="A set of exit codes that are allowed to be considered successful, separated by commas, such as '0,2'")
    sp.add_argument("--timeout", type=float, default=None, help="The timeout in seconds for each task. If the timeout is exceeded, the task will be killed and rc=124 will be recorded.")
    sp.set_defaults(func=cmd_run)

    sp = sub.add_parser("status", help="View progress summary")
    sp.add_argument("--db", required=True)
    sp.set_defaults(func=cmd_status)

    sp = sub.add_parser("export", help="Export as 0|cmd / 1|cmd text")
    sp.add_argument("--db", required=True)
    sp.add_argument("--out", required=True)
    sp.add_argument("--only-status", type=int, choices=[0,1,2], help="Export only specified states")
    sp.set_defaults(func=cmd_export)

    sp = sub.add_parser("failed", help="List failed attempts (status=0 and attempts>0)")
    sp.add_argument("--db", required=True)
    sp.set_defaults(func=cmd_failed)

    sp = sub.add_parser("recover", help="Reclaim stuck running tasks as queued")
    sp.add_argument("--db", required=True)
    sp.add_argument("--stale-seconds", type=int, default=3*3600, help="if 0<last_start stale")
    sp.set_defaults(func=cmd_recover)

    args = ap.parse_args()

    # if the user set strict_bash, enforce use_shell
    if getattr(args, "strict_bash", False) and not getattr(args, "use_shell", False):
        args.use_shell = True

    args.func(args)

if __name__ == "__main__":
    main()
