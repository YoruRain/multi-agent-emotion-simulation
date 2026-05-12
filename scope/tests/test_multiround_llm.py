from __future__ import annotations

import asyncio
import sys
import unittest
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from simulation.agent_loader import AgentRecord
from simulation.agent_state import AgentState
from simulation.multiround_config import MultiRoundSimulationConfig
from simulation.multiround_simulator import MultiRoundSimulator


def make_state(
    agent_id: str,
    influence_score: float,
    activity_score: float,
    propagation_role: str = "",
) -> AgentState:
    return AgentState(
        run_id="test-run",
        event_id="event-1",
        weibo_id="weibo-1",
        topic="测试事件",
        agent_id=agent_id,
        user_id=agent_id,
        round_id=0,
        memory_user_level="core",
        verified_type_name="",
        propagation_role=propagation_role,
        influence_level="",
        influence_score=influence_score,
        susceptibility_score=0.5,
        activity_score=activity_score,
        kol_sensitivity_score=0.5,
        media_dependency_score=0.5,
        repost_tendency_score=0.2,
        emotion_score=0.0,
        stance_score=0.0,
    )


class MultiRoundLLMTest(unittest.TestCase):
    def test_llm_budget_prioritizes_kol_then_score(self) -> None:
        simulator = MultiRoundSimulator(
            MultiRoundSimulationConfig(event_id="event-1", use_llm=True, max_llm_agents_per_round=2),
        )
        kol = make_state("kol", 0.2, 0.2)
        high_regular = make_state("high", 0.9, 0.9)
        mid_regular = make_state("mid", 0.6, 0.6)

        selected = simulator._select_llm_agent_ids(
            [
                (mid_regular, "regular_agent"),
                (high_regular, "regular_agent"),
                (kol, "kol_speaker"),
            ],
        )

        self.assertEqual(selected, {"kol", "high"})

    def test_llm_budget_zero_disables_calls(self) -> None:
        simulator = MultiRoundSimulator(
            MultiRoundSimulationConfig(event_id="event-1", use_llm=True, max_llm_agents_per_round=0),
        )
        selected = simulator._select_llm_agent_ids([(make_state("agent", 1.0, 1.0), "regular_agent")])

        self.assertEqual(selected, set())

    def test_missing_api_key_falls_back_for_budgeted_agent(self) -> None:
        simulator = MultiRoundSimulator(
            MultiRoundSimulationConfig(event_id="event-1", use_llm=True, max_llm_agents_per_round=None),
        )
        simulator.llm_generator.api_key = ""
        state = make_state("agent", 0.8, 0.8)
        agent = AgentRecord(
            agent_id="agent",
            user_id="agent",
            profile={"behavior_parameters": {}},
            memories=[],
            sys_prompt="你是一个微博用户。",
            memory_user_level="core",
        )

        result = asyncio.run(
            simulator._generate_reaction_for_state(
                state,
                agent,
                {"event_id": "event-1", "topic": "测试事件"},
                round_id=1,
                is_active=True,
                use_llm_for_agent=True,
            ),
        )

        self.assertEqual(result.source, "llm_fallback")
        self.assertTrue(result.llm_attempted)
        self.assertEqual(result.parse_status, "failed")
        self.assertTrue(result.reaction["participate"])


if __name__ == "__main__":
    unittest.main()
