"""End-to-end chosen/rejected conversation generation loop."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Callable, Sequence

from discoverllm.config import DEFAULT_CONFIG
from discoverllm.llm.base import LLMAdapter, LLMRequest, LLMResponse
from discoverllm.parsers.contracts import parse_synthesis_assistant_output
from discoverllm.prompts.loader import PromptLoader
from discoverllm.simulator import (
    AssistantResponseEvaluationRun,
    AssistantResponseEvaluator,
    StateUpdateResult,
    UserResponseGenerationRun,
    UserResponseGenerator,
    apply_evaluation_result,
    compute_turn_reward_from_state_update,
    select_active_subtree,
)
from discoverllm.types import (
    ConversationMessage,
    ConversationState,
    PreferenceExample,
    TrajectoryExample,
    TrajectoryTurn,
    TurnReward,
)

AssistantOutputParser = Callable[[str], dict[str, Any]]
TokenCounter = Callable[[str], int]


@dataclass(frozen=True, slots=True)
class AssistantCandidateGenerationRun:
    """Audit-friendly record of one assistant-candidate generation call."""

    candidate_name: str
    prompt_messages_json: list[dict[str, str]]
    system_prompt_name: str | None
    system_prompt_text: str | None
    llm_text_output: str
    llm_raw_output: str | dict[str, Any]
    parsed_output: dict[str, Any]
    assistant_message: ConversationMessage
    response_tokens: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.candidate_name, str) or not self.candidate_name.strip():
            raise ValueError("candidate_name must be a non-empty string.")
        if self.system_prompt_name is not None:
            if not isinstance(self.system_prompt_name, str) or not self.system_prompt_name.strip():
                raise ValueError("system_prompt_name must be a non-empty string or None.")
        if self.system_prompt_text is not None and not isinstance(self.system_prompt_text, str):
            raise TypeError("system_prompt_text must be a string or None.")
        if not isinstance(self.llm_text_output, str) or not self.llm_text_output.strip():
            raise ValueError("llm_text_output must be a non-empty string.")
        if not isinstance(self.llm_raw_output, (str, dict)):
            raise TypeError("llm_raw_output must be a string or dictionary.")
        if not isinstance(self.parsed_output, dict):
            raise TypeError("parsed_output must be a dictionary.")
        if not isinstance(self.assistant_message, ConversationMessage):
            raise TypeError("assistant_message must be a ConversationMessage.")
        if self.assistant_message.role != "assistant":
            raise ValueError("assistant_message.role must be 'assistant'.")
        if self.response_tokens is not None:
            if not isinstance(self.response_tokens, int):
                raise TypeError("response_tokens must be an int or None.")
            if self.response_tokens < 0:
                raise ValueError("response_tokens must be non-negative.")


@dataclass(frozen=True, slots=True)
class AssistantCandidateTurnRun:
    """Full per-candidate trace for one simulated assistant turn."""

    candidate_name: str
    generation_run: AssistantCandidateGenerationRun
    evaluation_run: AssistantResponseEvaluationRun
    state_update: StateUpdateResult
    reward: TurnReward

    def __post_init__(self) -> None:
        if not isinstance(self.candidate_name, str) or not self.candidate_name.strip():
            raise ValueError("candidate_name must be a non-empty string.")
        if not isinstance(self.generation_run, AssistantCandidateGenerationRun):
            raise TypeError("generation_run must be an AssistantCandidateGenerationRun.")
        if self.generation_run.candidate_name != self.candidate_name:
            raise ValueError("generation_run.candidate_name must match candidate_name.")
        if not isinstance(self.evaluation_run, AssistantResponseEvaluationRun):
            raise TypeError("evaluation_run must be an AssistantResponseEvaluationRun.")
        if not isinstance(self.state_update, StateUpdateResult):
            raise TypeError("state_update must be a StateUpdateResult.")
        if not isinstance(self.reward, TurnReward):
            raise TypeError("reward must be a TurnReward.")


@dataclass(frozen=True, slots=True)
class SimulatedConversationTurn:
    """One chosen/rejected comparison turn in the conversation loop."""

    turn_index: int
    prompt_messages: tuple[ConversationMessage, ...]
    candidate_runs: tuple[AssistantCandidateTurnRun, AssistantCandidateTurnRun]
    chosen_candidate_name: str
    rejected_candidate_name: str
    next_user_generation: UserResponseGenerationRun | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.turn_index, int):
            raise TypeError("turn_index must be an int.")
        if self.turn_index < 0:
            raise ValueError("turn_index must be non-negative.")
        if not isinstance(self.prompt_messages, tuple) or not self.prompt_messages:
            raise ValueError("prompt_messages must be a non-empty tuple.")
        for message in self.prompt_messages:
            if not isinstance(message, ConversationMessage):
                raise TypeError("prompt_messages must contain ConversationMessage entries.")
        if len(self.candidate_runs) != 2:
            raise ValueError("candidate_runs must contain exactly two assistant candidates.")
        for run in self.candidate_runs:
            if not isinstance(run, AssistantCandidateTurnRun):
                raise TypeError("candidate_runs must contain AssistantCandidateTurnRun entries.")
        names = {run.candidate_name for run in self.candidate_runs}
        if self.chosen_candidate_name not in names:
            raise ValueError("chosen_candidate_name must match one candidate run.")
        if self.rejected_candidate_name not in names:
            raise ValueError("rejected_candidate_name must match one candidate run.")
        if self.chosen_candidate_name == self.rejected_candidate_name:
            raise ValueError("chosen_candidate_name and rejected_candidate_name must differ.")
        if self.next_user_generation is not None and not isinstance(
            self.next_user_generation, UserResponseGenerationRun
        ):
            raise TypeError("next_user_generation must be a UserResponseGenerationRun or None.")


@dataclass(frozen=True, slots=True)
class ConversationScopeAudit:
    """Conversation-level summary of which root subtrees were ever reached."""

    active_root_history: tuple[str, ...]
    ever_active_root_ids: tuple[str, ...]
    never_active_root_ids: tuple[str, ...]
    next_active_root_id: str | None

    def __post_init__(self) -> None:
        for field_value in (self.active_root_history, self.ever_active_root_ids, self.never_active_root_ids):
            if not isinstance(field_value, tuple):
                raise TypeError("scope-audit root collections must be tuples.")
            for item in field_value:
                if not isinstance(item, str) or not item.strip():
                    raise ValueError("scope-audit root ids must be non-empty strings.")
        if self.next_active_root_id is not None:
            if not isinstance(self.next_active_root_id, str) or not self.next_active_root_id.strip():
                raise ValueError("next_active_root_id must be a non-empty string or None.")


@dataclass(frozen=True, slots=True)
class ConversationSimulationResult:
    """Complete paper-faithful simulation output for one artifact."""

    trajectory: TrajectoryExample
    preferences: list[PreferenceExample]
    turn_runs: list[SimulatedConversationTurn]
    final_state: ConversationState
    stop_reason: str
    scope_audit: ConversationScopeAudit

    def __post_init__(self) -> None:
        if not isinstance(self.trajectory, TrajectoryExample):
            raise TypeError("trajectory must be a TrajectoryExample.")
        if not isinstance(self.preferences, list):
            raise TypeError("preferences must be a list of PreferenceExample.")
        for item in self.preferences:
            if not isinstance(item, PreferenceExample):
                raise TypeError("preferences must contain PreferenceExample entries.")
        if not isinstance(self.turn_runs, list):
            raise TypeError("turn_runs must be a list of SimulatedConversationTurn.")
        for item in self.turn_runs:
            if not isinstance(item, SimulatedConversationTurn):
                raise TypeError("turn_runs must contain SimulatedConversationTurn entries.")
        if not isinstance(self.final_state, ConversationState):
            raise TypeError("final_state must be a ConversationState.")
        if not isinstance(self.stop_reason, str) or not self.stop_reason.strip():
            raise ValueError("stop_reason must be a non-empty string.")
        if not isinstance(self.scope_audit, ConversationScopeAudit):
            raise TypeError("scope_audit must be a ConversationScopeAudit.")


@dataclass(slots=True)
class _CandidateBranchResult:
    turn_run: AssistantCandidateTurnRun
    branch_state: ConversationState


class AssistantCandidateGenerator:
    """Pluggable assistant candidate generator with optional system prompt."""

    def __init__(
        self,
        *,
        candidate_name: str,
        llm_adapter: LLMAdapter,
        prompt_loader: PromptLoader | None = None,
        system_prompt_name: str | None = None,
        output_parser: AssistantOutputParser | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> None:
        if not isinstance(candidate_name, str) or not candidate_name.strip():
            raise ValueError("candidate_name must be a non-empty string.")
        if not isinstance(llm_adapter, LLMAdapter):
            raise TypeError("llm_adapter must implement LLMAdapter.")
        if prompt_loader is None:
            prompt_loader = PromptLoader()
        if not isinstance(prompt_loader, PromptLoader):
            raise TypeError("prompt_loader must be a PromptLoader.")
        if system_prompt_name is not None:
            if not isinstance(system_prompt_name, str) or not system_prompt_name.strip():
                raise ValueError("system_prompt_name must be a non-empty string or None.")
        if not isinstance(temperature, (int, float)):
            raise TypeError("temperature must be numeric.")
        temperature = float(temperature)
        if temperature < 0:
            raise ValueError("temperature must be non-negative.")
        if max_tokens is not None:
            if not isinstance(max_tokens, int):
                raise TypeError("max_tokens must be an int or None.")
            if max_tokens <= 0:
                raise ValueError("max_tokens must be positive when provided.")

        self._candidate_name = candidate_name.strip()
        self._llm = llm_adapter
        self._prompts = prompt_loader
        self._system_prompt_name = system_prompt_name.strip() if system_prompt_name else None
        self._output_parser = output_parser or _default_output_parser(self._system_prompt_name)
        self._temperature = temperature
        self._max_tokens = max_tokens

    @property
    def candidate_name(self) -> str:
        return self._candidate_name

    def generate_response(
        self,
        messages: Sequence[ConversationMessage],
    ) -> AssistantCandidateGenerationRun:
        """Generate one assistant candidate message from the current chat history."""

        if not messages:
            raise ValueError("messages must not be empty.")
        prompt_messages = _copy_messages(messages)
        prompt_messages_json = _messages_to_payload(prompt_messages)
        system_prompt_text = None
        if self._system_prompt_name is not None:
            system_prompt_text = self._prompts.load_template(self._system_prompt_name)

        response = self._generate(
            messages=prompt_messages,
            system_prompt_text=system_prompt_text,
        )
        parsed_output = self._output_parser(response.text)
        assistant_text = _extract_response_text(parsed_output, raw_text=response.text)
        assistant_message = ConversationMessage(role="assistant", content=assistant_text)
        return AssistantCandidateGenerationRun(
            candidate_name=self._candidate_name,
            prompt_messages_json=prompt_messages_json,
            system_prompt_name=self._system_prompt_name,
            system_prompt_text=system_prompt_text,
            llm_text_output=response.text,
            llm_raw_output=response.raw_output,
            parsed_output=parsed_output,
            assistant_message=assistant_message,
            response_tokens=response.completion_tokens,
        )

    def _generate(
        self,
        *,
        messages: Sequence[ConversationMessage],
        system_prompt_text: str | None,
    ) -> LLMResponse:
        request = LLMRequest(
            messages=messages,
            system_prompt=system_prompt_text,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
            metadata={
                "candidate_name": self._candidate_name,
                "system_prompt_name": self._system_prompt_name,
                "prompt_name": self._system_prompt_name or "assistant_candidate",
            },
        )
        return self._llm.generate(request)


class ConversationSimulator:
    """Run the paper's two-candidate chosen/rejected simulation loop."""

    def __init__(
        self,
        *,
        assistant_generators: Sequence[AssistantCandidateGenerator],
        evaluator: AssistantResponseEvaluator,
        user_generator: UserResponseGenerator,
        turn_limit: int = DEFAULT_CONFIG.default_turn_limit,
        tangential_probability: float = DEFAULT_CONFIG.tangential_probability,
        tau: int = DEFAULT_CONFIG.tau_sft_dpo,
        length_penalty_lambda: float = DEFAULT_CONFIG.length_penalty_lambda,
        token_counter: TokenCounter | None = None,
    ) -> None:
        if not isinstance(assistant_generators, Sequence):
            raise TypeError("assistant_generators must be a sequence.")
        generators = list(assistant_generators)
        if len(generators) != 2:
            raise ValueError("assistant_generators must contain exactly two candidates.")
        for generator in generators:
            if not isinstance(generator, AssistantCandidateGenerator):
                raise TypeError(
                    "assistant_generators must contain AssistantCandidateGenerator entries."
                )
        if generators[0].candidate_name == generators[1].candidate_name:
            raise ValueError("assistant candidate names must be distinct.")
        if not isinstance(evaluator, AssistantResponseEvaluator):
            raise TypeError("evaluator must be an AssistantResponseEvaluator.")
        if not isinstance(user_generator, UserResponseGenerator):
            raise TypeError("user_generator must be a UserResponseGenerator.")
        if not isinstance(turn_limit, int):
            raise TypeError("turn_limit must be an int.")
        if turn_limit <= 0:
            raise ValueError("turn_limit must be positive.")
        if not isinstance(tangential_probability, (int, float)):
            raise TypeError("tangential_probability must be numeric.")
        tangential_probability = float(tangential_probability)
        if not 0.0 <= tangential_probability <= 1.0:
            raise ValueError("tangential_probability must be in [0, 1].")
        if not isinstance(tau, int):
            raise TypeError("tau must be an int.")
        if tau <= 0:
            raise ValueError("tau must be positive.")
        if not isinstance(length_penalty_lambda, (int, float)):
            raise TypeError("length_penalty_lambda must be numeric.")
        length_penalty_lambda = float(length_penalty_lambda)
        if length_penalty_lambda <= 0:
            raise ValueError("length_penalty_lambda must be positive.")

        self._assistant_generators = tuple(generators)
        self._evaluator = evaluator
        self._user_generator = user_generator
        self._turn_limit = turn_limit
        self._tangential_probability = tangential_probability
        self._tau = tau
        self._length_penalty_lambda = length_penalty_lambda
        self._token_counter = token_counter

    def simulate_conversation(
        self,
        conversation_state: ConversationState,
    ) -> ConversationSimulationResult:
        """Run up to `turn_limit` assistant turns, keeping only the chosen branch."""

        if not isinstance(conversation_state, ConversationState):
            raise TypeError("conversation_state must be a ConversationState.")

        canonical_state = copy.deepcopy(conversation_state)
        trajectory_turns: list[TrajectoryTurn] = []
        preferences: list[PreferenceExample] = []
        turn_runs: list[SimulatedConversationTurn] = []
        stop_reason = "turn_limit_reached"

        for _ in range(self._turn_limit):
            if not canonical_state.messages:
                raise ValueError("conversation_state.messages must not be empty.")
            if canonical_state.messages[-1].role != "user":
                raise ValueError(
                    "Conversation state must end with a user message before assistant turns."
                )
            if select_active_subtree(canonical_state.intent_forest) is None:
                stop_reason = "no_active_subtree"
                break

            prompt_messages = _copy_messages(canonical_state.messages)
            branch_results = [
                self._run_candidate_branch(generator, canonical_state, prompt_messages)
                for generator in self._assistant_generators
            ]
            chosen_index, rejected_index = _choose_candidate_indices(branch_results)
            chosen = branch_results[chosen_index]
            rejected = branch_results[rejected_index]

            chosen_state = chosen.branch_state
            current_turn_index = canonical_state.turn_index
            chosen_state.turn_index = current_turn_index + 1

            next_user_generation = None
            if chosen_state.turn_index >= self._turn_limit:
                stop_reason = "turn_limit_reached"
            elif select_active_subtree(chosen_state.intent_forest) is None:
                stop_reason = "no_active_subtree"
            else:
                try:
                    next_user_generation = self._user_generator.generate_next_message(chosen_state)
                except ValueError as exc:
                    if str(exc) != "No unsatisfied intents remain for user response generation.":
                        raise
                    stop_reason = "all_goals_satisfied"
                else:
                    chosen_state.messages.append(next_user_generation.emitted_message)

            turn_runs.append(
                SimulatedConversationTurn(
                    turn_index=current_turn_index,
                    prompt_messages=tuple(prompt_messages),
                    candidate_runs=(
                        branch_results[0].turn_run,
                        branch_results[1].turn_run,
                    ),
                    chosen_candidate_name=chosen.turn_run.candidate_name,
                    rejected_candidate_name=rejected.turn_run.candidate_name,
                    next_user_generation=next_user_generation,
                )
            )
            trajectory_turns.append(
                TrajectoryTurn(
                    turn_index=current_turn_index,
                    user_message=prompt_messages[-1].content,
                    chosen_assistant_message=chosen.turn_run.generation_run.assistant_message.content,
                    rejected_assistant_message=(
                        rejected.turn_run.generation_run.assistant_message.content
                    ),
                    chosen_reward=chosen.turn_run.reward,
                    rejected_reward=rejected.turn_run.reward,
                )
            )
            preferences.append(
                PreferenceExample(
                    artifact_id=canonical_state.artifact.artifact_id,
                    turn_index=current_turn_index,
                    prompt_messages=_copy_messages(prompt_messages),
                    chosen_response=chosen.turn_run.generation_run.assistant_message.content,
                    rejected_response=rejected.turn_run.generation_run.assistant_message.content,
                    chosen_reward=chosen.turn_run.reward,
                    rejected_reward=rejected.turn_run.reward,
                )
            )

            canonical_state = chosen_state
            if next_user_generation is None:
                break

        final_state = copy.deepcopy(canonical_state)
        trajectory = TrajectoryExample(
            artifact_id=final_state.artifact.artifact_id,
            turns=trajectory_turns,
            messages=_copy_messages(final_state.messages),
            final_state=copy.deepcopy(final_state),
            seed=final_state.random_seed,
        )
        scope_audit = _build_conversation_scope_audit(
            turn_runs,
            final_state=final_state,
        )
        return ConversationSimulationResult(
            trajectory=trajectory,
            preferences=preferences,
            turn_runs=turn_runs,
            final_state=final_state,
            stop_reason=stop_reason,
            scope_audit=scope_audit,
        )

    def _run_candidate_branch(
        self,
        generator: AssistantCandidateGenerator,
        canonical_state: ConversationState,
        prompt_messages: Sequence[ConversationMessage],
    ) -> _CandidateBranchResult:
        branch_state = copy.deepcopy(canonical_state)
        generation_run = generator.generate_response(prompt_messages)
        branch_state.messages.append(generation_run.assistant_message)
        evaluation_run = self._evaluator.evaluate_messages(
            branch_state.messages,
            branch_state.intent_forest,
        )
        state_update = apply_evaluation_result(
            branch_state,
            evaluation_run.evaluation,
            tangential_probability=self._tangential_probability,
        )
        reward = compute_turn_reward_from_state_update(
            state_update,
            assistant_message=generation_run.assistant_message.content,
            response_tokens=generation_run.response_tokens,
            tau=self._tau,
            length_penalty_lambda=self._length_penalty_lambda,
            token_counter=self._token_counter,
        )
        return _CandidateBranchResult(
            turn_run=AssistantCandidateTurnRun(
                candidate_name=generator.candidate_name,
                generation_run=generation_run,
                evaluation_run=evaluation_run,
                state_update=state_update,
                reward=reward,
            ),
            branch_state=branch_state,
        )


def _choose_candidate_indices(
    branch_results: Sequence[_CandidateBranchResult],
) -> tuple[int, int]:
    if len(branch_results) != 2:
        raise ValueError("branch_results must contain exactly two candidate results.")
    first_reward = branch_results[0].turn_run.reward.total_reward
    second_reward = branch_results[1].turn_run.reward.total_reward
    if first_reward >= second_reward:
        return 0, 1
    return 1, 0


def _default_output_parser(system_prompt_name: str | None) -> AssistantOutputParser:
    if system_prompt_name == "synthesis_assistant":
        return parse_synthesis_assistant_output
    return parse_plain_assistant_output


def parse_plain_assistant_output(raw_text: str) -> dict[str, str]:
    """Treat the raw model output as the assistant reply."""

    if not isinstance(raw_text, str) or not raw_text.strip():
        raise ValueError("assistant output must be a non-empty string.")
    return {"response": raw_text.strip()}


def _extract_response_text(parsed_output: dict[str, Any], *, raw_text: str) -> str:
    response = parsed_output.get("response")
    if not isinstance(response, str) or not response.strip():
        raise ValueError("assistant parsed output must contain a non-empty 'response' field.")
    return response


def _copy_messages(messages: Sequence[ConversationMessage]) -> list[ConversationMessage]:
    copied: list[ConversationMessage] = []
    for message in messages:
        if not isinstance(message, ConversationMessage):
            raise TypeError("messages must contain ConversationMessage entries.")
        copied.append(ConversationMessage(role=message.role, content=message.content))
    return copied


def _messages_to_payload(messages: Sequence[ConversationMessage]) -> list[dict[str, str]]:
    return [{"role": message.role, "content": message.content} for message in messages]


def _build_conversation_scope_audit(
    turn_runs: Sequence[SimulatedConversationTurn],
    *,
    final_state: ConversationState,
) -> ConversationScopeAudit:
    active_root_history = tuple(
        turn.candidate_runs[0].evaluation_run.active_root_id for turn in turn_runs
    )
    seen: set[str] = set()
    ever_active_root_ids: list[str] = []
    for root_id in active_root_history:
        if root_id in seen:
            continue
        seen.add(root_id)
        ever_active_root_ids.append(root_id)

    never_active_root_ids = tuple(
        tree.root.id
        for tree in final_state.intent_forest.trees
        if tree.root.id not in seen
    )
    next_active_tree = select_active_subtree(final_state.intent_forest)
    return ConversationScopeAudit(
        active_root_history=active_root_history,
        ever_active_root_ids=tuple(ever_active_root_ids),
        never_active_root_ids=never_active_root_ids,
        next_active_root_id=next_active_tree.root.id if next_active_tree is not None else None,
    )
