#!/usr/bin/env python3
import json, hashlib
from pathlib import Path
from collections import defaultdict
import yaml

RUNS = Path("/homes/sohawan2/reinforcement-learning-thesis/thesis_project/runs_seed0")
OUT  = Path("/homes/sohawan2/reinforcement-learning-thesis/thesis_project/audit_config_out")
OUT.mkdir(parents=True, exist_ok=True)

def stable_json(obj) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",",":"))

def key_from_cfg(cfg: dict) -> str:
    # 唯一标识一个超参组合（建议用 parsed）
    env = cfg.get("env_id")
    algo = cfg.get("algo")
    seed = cfg.get("seed")
    hp = cfg.get("hyperparams_parsed", cfg.get("hyperparams_raw", None))
    # policy_kwargs 等嵌套结构也 OK
    payload = {"env_id": env, "algo": algo, "seed": seed, "hyperparams": hp}
    h = hashlib.sha256(stable_json(payload).encode("utf-8")).hexdigest()[:16]
    return f"{env}__{algo}__seed{seed}__{h}"

def build_cmd_from_cfg(cfg: dict) -> str:
    # 尽量还原 runner.py 命令（贴近你 DB cmd 的风格）
    env = cfg.get("env_id")
    algo = cfg.get("algo")
    seed = cfg.get("seed")
    hp_raw = cfg.get("hyperparams_raw", [])
    # 你的 DB 风格：--hyperparams k:v k:v ... 其中 policy_kwargs:dict(...)
    hp_str = " ".join([str(x).strip() for x in hp_raw if str(x).strip()])
    # 注意：policy_kwargs 那种含括号/逗号的，最好整体加引号（你 DB 里就是这么做的）
    # 这里做个简单处理：如果包含空格或括号，外面加单引号
    tokens=[]
    for item in hp_raw:
        s=str(item).strip()
        if not s: 
            continue
        if any(ch in s for ch in [" ", "(", ")", "[", "]", "{", "}", ","]):
            # 避免已有引号重复
            if not ((s.startswith("'") and s.endswith("'")) or (s.startswith('"') and s.endswith('"'))):
                s = "'" + s.replace("'", "\\'") + "'"
        tokens.append(s)
    hp_cli = " ".join(tokens)

    return f"python runner.py --algo {algo} --env {env} --hyperparams {hp_cli} --seed {seed}"

def main():
    # 收集所有 run dirs（含 config.yml）
    run_entries = []
    for cfg_path in RUNS.rglob("config.yml"):
        run_dir = cfg_path.parent
        res_path = run_dir / "results.json"
        try:
            cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
        except Exception as e:
            run_entries.append({
                "run_dir": str(run_dir),
                "config_path": str(cfg_path),
                "error": f"YAML read failed: {e}",
                "has_results": res_path.exists()
            })
            continue

        k = key_from_cfg(cfg)
        run_entries.append({
            "key": k,
            "run_dir": str(run_dir),
            "config_path": str(cfg_path),
            "has_results": res_path.exists(),
            "timestamp": cfg.get("timestamp"),
            "pid": cfg.get("pid"),
            "env_id": cfg.get("env_id"),
            "algo": cfg.get("algo"),
            "seed": cfg.get("seed"),
            "hyperparams_parsed": cfg.get("hyperparams_parsed"),
            "hyperparams_raw": cfg.get("hyperparams_raw"),
        })

    # 分组：同一个 key 可能有多个 run_dir
    groups = defaultdict(list)
    bad = []
    for e in run_entries:
        if "key" not in e:
            bad.append(e)
        else:
            groups[e["key"]].append(e)

    # 对每组：挑保留项
    keep = {}
    dup_complete = []
    dup_incomplete = []
    need_rerun_keys = []

    def sort_score(e):
        # 优先保留有结果的；其次按 timestamp 字符串排序（越新越大）
        return (1 if e["has_results"] else 0, e.get("timestamp") or "", e.get("pid") or -1)

    for k, items in groups.items():
        items_sorted = sorted(items, key=sort_score, reverse=True)
        keep_item = items_sorted[0]
        keep[k] = keep_item

        # 统计重复
        if len(items_sorted) > 1:
            for it in items_sorted[1:]:
                if it["has_results"]:
                    dup_complete.append(it)
                else:
                    dup_incomplete.append(it)

        # 判断是否需要重跑：该 key 下是否存在任何 has_results
        if not any(it["has_results"] for it in items):
            need_rerun_keys.append({
                "key": k,
                "env_id": keep_item.get("env_id"),
                "algo": keep_item.get("algo"),
                "seed": keep_item.get("seed"),
                "example_run_dir": keep_item.get("run_dir"),
                "example_config_path": keep_item.get("config_path"),
                "hyperparams_raw": keep_item.get("hyperparams_raw"),
                "hyperparams_parsed": keep_item.get("hyperparams_parsed"),
            })

    # 输出文件
    OUT.joinpath("summary.json").write_text(json.dumps({
        "runs_root": str(RUNS),
        "total_run_dirs_with_config": len(groups),
        "total_config_files_found": len(run_entries),
        "bad_config_count": len(bad),
        "duplicate_keys_count": sum(1 for k,v in groups.items() if len(v)>1),
        "dup_complete_dirs": len(dup_complete),
        "dup_incomplete_dirs": len(dup_incomplete),
        "need_rerun_keys": len(need_rerun_keys),
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    # lists
    OUT.joinpath("duplicate_incomplete_dirs.txt").write_text(
        "\n".join(e["run_dir"] for e in dup_incomplete) + ("\n" if dup_incomplete else ""),
        encoding="utf-8"
    )
    OUT.joinpath("duplicate_complete_dirs.txt").write_text(
        "\n".join(e["run_dir"] for e in dup_complete) + ("\n" if dup_complete else ""),
        encoding="utf-8"
    )

    # need rerun
    with OUT.joinpath("need_rerun_keys.jsonl").open("w", encoding="utf-8") as f:
        for x in need_rerun_keys:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")

    # 生成 rerun cmds（基于示例 cfg）
    with OUT.joinpath("rerun_cmds.sh").open("w", encoding="utf-8") as f:
        f.write("#!/usr/bin/env bash\nset -euo pipefail\n\n")
        for x in need_rerun_keys:
            cmd = build_cmd_from_cfg({
                "env_id": x["env_id"],
                "algo": x["algo"],
                "seed": x["seed"],
                "hyperparams_raw": x["hyperparams_raw"] or [],
            })
            f.write(cmd + "\n")
    OUT.joinpath("rerun_cmds.sh").chmod(0o755)

    # 生成 move-to-trash 脚本（只移动重复且不完整的，最安全）
    trash = RUNS.parent / (RUNS.name + "__trash")
    with OUT.joinpath("move_dup_incomplete_to_trash.sh").open("w", encoding="utf-8") as f:
        f.write("#!/usr/bin/env bash\nset -euo pipefail\n")
        f.write(f"mkdir -p '{trash}'\n")
        for rd in dup_incomplete:
            rd_p = Path(rd["run_dir"])
            try:
                rel = rd_p.relative_to(RUNS)
                dest = trash / rel
            except Exception:
                dest = trash / rd_p.name
            f.write(f"mkdir -p '{dest.parent}'\n")
            f.write(f"mv '{rd_p}' '{dest}'\n")
    OUT.joinpath("move_dup_incomplete_to_trash.sh").chmod(0o755)

    print("Wrote outputs to:", OUT)
    print(OUT.joinpath("summary.json").read_text(encoding="utf-8"))

if __name__ == "__main__":
    main()
