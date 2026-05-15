from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from .agent_state import AgentState
from .interaction_schema import InteractionRecord

LOGGER = logging.getLogger(__name__)


def _graph_value(value: Any) -> str | int | float | bool:
    if value is None:
        return ""
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def build_interaction_graph(
    agent_states: list[AgentState],
    interactions: list[InteractionRecord],
):
    try:
        import networkx as nx
    except ImportError as exc:
        raise RuntimeError("networkx is required to build network.graphml. Please install networkx.") from exc

    graph = nx.DiGraph()
    latest_states: dict[str, AgentState] = {}
    for state in agent_states:
        current = latest_states.get(state.agent_id)
        if current is None or state.round_id >= current.round_id:
            latest_states[state.agent_id] = state

    for state in latest_states.values():
        graph.add_node(
            state.agent_id,
            agent_id=state.agent_id,
            user_id=_graph_value(state.user_id),
            memory_user_level=_graph_value(state.memory_user_level),
            verified_type_name=_graph_value(state.verified_type_name),
            propagation_role=_graph_value(state.propagation_role),
            influence_level=_graph_value(state.influence_level),
            influence_score=state.influence_score,
            susceptibility_score=state.susceptibility_score,
            activity_score=state.activity_score,
            kol_sensitivity_score=state.kol_sensitivity_score,
            media_dependency_score=state.media_dependency_score,
            final_emotion_score=state.emotion_score,
            final_stance_score=state.stance_score,
            final_emotion_label=state.emotion_label,
            final_stance_label=state.stance_label,
        )

    for interaction in interactions:
        source = interaction.source_agent_id
        target = interaction.target_agent_id
        if not graph.has_node(source):
            graph.add_node(source, agent_id=source)
        if not graph.has_node(target):
            graph.add_node(target, agent_id=target)
        if graph.has_edge(source, target):
            edge = graph[source][target]
            edge["weight_sum"] = round(float(edge.get("weight_sum", 0.0)) + interaction.weight, 4)
            edge["interaction_count"] = int(edge.get("interaction_count", 0)) + 1
            edge["last_round"] = max(int(edge.get("last_round", interaction.round_id)), interaction.round_id)
            edge["weight"] = round(edge["weight_sum"] / edge["interaction_count"], 4)
            types = set(str(edge.get("interaction_types", "")).split(","))
            types.add(interaction.interaction_type)
            edge["interaction_types"] = ",".join(sorted(token for token in types if token))
            edge["interaction_type"] = edge["interaction_types"]
            edge["target_action_type"] = _graph_value(interaction.target_action_type)
        else:
            graph.add_edge(
                source,
                target,
                weight=interaction.weight,
                weight_sum=interaction.weight,
                interaction_count=1,
                first_round=interaction.round_id,
                last_round=interaction.round_id,
                round_id=interaction.round_id,
                interaction_type=interaction.interaction_type,
                interaction_types=interaction.interaction_type,
                source_action_type=interaction.source_action_type,
                target_action_type=_graph_value(interaction.target_action_type),
            )

    return graph


def save_graphml(graph, output_path: Path) -> None:
    try:
        import networkx as nx
    except ImportError as exc:
        raise RuntimeError("networkx is required to save network.graphml. Please install networkx.") from exc

    output_path.parent.mkdir(parents=True, exist_ok=True)
    nx.write_graphml(graph, output_path)
    LOGGER.info("Wrote interaction graph to %s", output_path)
