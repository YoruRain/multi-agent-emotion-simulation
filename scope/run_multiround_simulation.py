from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "scope" / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from simulation.multiround_config import DEFAULT_MULTIROUND_OUTPUT_DIR, MultiRoundSimulationConfig  # noqa: E402
from simulation.multiround_simulator import MultiRoundSimulator  # noqa: E402
from simulation.single_event_simulator import configure_logging  # noqa: E402


def _parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    lowered = value.strip().lower()
    if lowered in {"true", "1", "yes", "y"}:
        return True
    if lowered in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a multi-round Weibo user state simulation skeleton.")
    parser.add_argument("--event-id", required=True, help="Target event_id in events.jsonl.")
    parser.add_argument("--max-agents", type=int, default=None, help="Run at most N agents after filtering.")
    parser.add_argument("--memory-user-level", default=None, help="Optional memory_user_level filter, e.g. core.")
    parser.add_argument("--rounds", type=int, default=5, help="Number of simulation rounds after round 0.")
    parser.add_argument("--active-agent-limit", type=int, default=None, help="Max active agents retained per round.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_MULTIROUND_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=42, help="Seed for agent sampling and participation decisions.")
    parser.add_argument("--use-llm", type=_parse_bool, default=False, help="Reserved; false uses fallback rules.")
    parser.add_argument("--max-llm-agents-per-round", type=int, default=None, help="Reserved for later LLM-enabled stages.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite an existing run directory if the generated id collides.")
    parser.add_argument("--resume", dest="resume", action="store_true", default=True, help="Reserved; kept for CLI compatibility.")
    parser.add_argument("--no-resume", dest="resume", action="store_false", help="Disable resume compatibility flag.")
    parser.add_argument("--dry-run", action="store_true", help="Load event and agents, then print first 3 initial states.")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging(args.verbose)
    config = MultiRoundSimulationConfig(
        event_id=args.event_id,
        max_agents=args.max_agents,
        memory_user_level=args.memory_user_level,
        rounds=args.rounds,
        active_agent_limit=args.active_agent_limit,
        use_llm=args.use_llm,
        max_llm_agents_per_round=args.max_llm_agents_per_round,
        seed=args.seed,
        output_dir=args.output_dir,
        overwrite=args.overwrite,
        resume=args.resume,
        dry_run=args.dry_run,
    )
    simulator = MultiRoundSimulator(config)
    try:
        result = simulator.run()
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    if result.get("dry_run"):
        print(f"dry_run: true")
        print(f"run_id: {result['run_id']}")
        print(f"agent_count: {result['agent_count']}")
        print(f"rounds: {result['rounds']}")
        return

    print(f"run_id: {result['run_id']}")
    print(f"output_dir: {result['output_dir']}")
    print(f"agent_count: {result['agent_count']}")
    print(f"rounds: {result['rounds']}")
    print(f"final_avg_emotion_score: {result['final_avg_emotion_score']}")
    print(f"final_avg_stance_score: {result['final_avg_stance_score']}")
    print(f"round_metrics_csv: {result['round_metrics_path']}")


if __name__ == "__main__":
    main()
