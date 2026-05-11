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

from simulation.single_event_simulator import (  # noqa: E402
    DEFAULT_OUTPUT_DIR,
    SimulatorPaths,
    SingleEventSimulator,
    configure_logging,
    run_event_sync,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a single-event AgentScope Weibo user simulation.")
    parser.add_argument("--event-id", required=True, help="Target event_id in data/scope/events.jsonl.")
    parser.add_argument("--max-agents", type=int, default=None, help="Run at most N agents after filtering.")
    parser.add_argument("--memory-user-level", default=None, help="Optional memory_user_level filter, e.g. core.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing run output file.")
    parser.add_argument("--resume", dest="resume", action="store_true", default=True, help="Skip completed event_id + agent_id rows.")
    parser.add_argument("--no-resume", dest="resume", action="store_false", help="Do not skip completed rows.")
    parser.add_argument("--dry-run", action="store_true", help="Inspect prompts and messages without calling a model.")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--model-name", default=None, help="Override MODEL_NAME.")
    parser.add_argument("--base-url", default=None, help="Override BASE_URL.")
    parser.add_argument("--profiles-path", type=Path, default=None)
    parser.add_argument("--memories-path", type=Path, default=None)
    parser.add_argument("--sys-prompts-path", type=Path, default=None)
    parser.add_argument("--events-path", type=Path, default=None)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging(args.verbose)

    paths = SimulatorPaths(
        profiles_path=args.profiles_path or SimulatorPaths.profiles_path,
        memories_path=args.memories_path or SimulatorPaths.memories_path,
        sys_prompts_path=args.sys_prompts_path or SimulatorPaths.sys_prompts_path,
        events_path=args.events_path or SimulatorPaths.events_path,
        output_dir=args.output_dir,
    )
    simulator = SingleEventSimulator(
        paths=paths,
        model_name=args.model_name,
        base_url=args.base_url,
        seed=args.seed,
    )
    run_event_sync(
        simulator,
        event_id=args.event_id,
        max_agents=args.max_agents,
        memory_user_level=args.memory_user_level,
        output_dir=args.output_dir,
        overwrite=args.overwrite,
        resume=args.resume,
        dry_run=args.dry_run,
        concurrency=args.concurrency,
    )


if __name__ == "__main__":
    main()
