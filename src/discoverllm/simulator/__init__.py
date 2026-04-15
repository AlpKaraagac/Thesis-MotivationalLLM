"""Simulation modules for evaluator, state updates, and user responses."""

from discoverllm.simulator.evaluator import (
    AssistantResponseEvaluationRun,
    AssistantResponseEvaluator,
    EvaluationScopeAudit,
    build_evaluation_scope_audit,
    evaluation_result_to_payload,
    messages_to_payload,
    normalize_evaluation_result,
    select_active_subtree,
    subtree_to_payload,
)
from discoverllm.simulator.reward import (
    compute_efficiency_penalty,
    compute_turn_reward,
    compute_turn_reward_from_state_update,
    count_whitespace_tokens,
)
from discoverllm.simulator.state_updater import (
    StateUpdateResult,
    apply_evaluation_result,
)
from discoverllm.simulator.user_generator import (
    GoalStatusItem,
    GoalStatusView,
    UserResponseGenerationRun,
    UserResponseGenerator,
    build_goal_status_view,
)

__all__ = [
    "AssistantResponseEvaluationRun",
    "AssistantResponseEvaluator",
    "EvaluationScopeAudit",
    "GoalStatusItem",
    "GoalStatusView",
    "StateUpdateResult",
    "UserResponseGenerationRun",
    "UserResponseGenerator",
    "apply_evaluation_result",
    "build_evaluation_scope_audit",
    "build_goal_status_view",
    "compute_efficiency_penalty",
    "compute_turn_reward",
    "compute_turn_reward_from_state_update",
    "count_whitespace_tokens",
    "evaluation_result_to_payload",
    "messages_to_payload",
    "normalize_evaluation_result",
    "select_active_subtree",
    "subtree_to_payload",
]
