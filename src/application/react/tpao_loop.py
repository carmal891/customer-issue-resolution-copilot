"""
TPAO (Think-Plan-Act-Observe) Loop Implementation with LLM Integration

This is our concrete implementation of the ReAct reasoning pattern for novel task handling.
The loop iterates through phases until the issue is resolved or max iterations reached.

Phases:
1. Think: Analyze the issue and identify information needs using LLM
2. Plan: Draft resolution steps using RAG context and LLM reasoning
3. Act: Prepare actions for human approval
4. Observe: Record results and update context

This loop is used only for novel tasks where no existing skill matches.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime
import logging
from enum import Enum
import json

from src.domain.models.issue import Issue
from src.domain.models.resolution import Resolution, ResolutionStep, ResolutionStatus
from src.application.rag.rag_pipeline import RAGPipeline
from src.application.tools.base import ToolRegistry
from src.infrastructure.llm.llm_service import LLMService

logger = logging.getLogger(__name__)


class TPAOPhase(str, Enum):
    """TPAO loop phases."""
    THINK = "think"
    PLAN = "plan"
    ACT = "act"
    OBSERVE = "observe"
    COMPLETE = "complete"


@dataclass
class TPAOState:
    """State maintained across TPAO iterations."""
    issue: Issue
    resolution: Resolution
    current_phase: TPAOPhase
    iteration: int
    max_iterations: int
    context_history: List[str]
    action_history: List[Dict[str, Any]]
    observations: List[str]
    is_resolved: bool
    needs_approval: bool
    error: Optional[str] = None


class TPAOLoop:
    """
    TPAO Loop for novel task handling with LLM integration.
    
    This implements the ReAct-style reasoning pattern:
    - Think: What do I need to know? (LLM analyzes issue)
    - Plan: What steps should I take? (LLM generates plan)
    - Act: What actions need approval? (Convert to ResolutionSteps)
    - Observe: What were the results? (Record and decide next iteration)
    """
    
    def __init__(
        self,
        rag_pipeline: RAGPipeline,
        tool_registry: ToolRegistry,
        llm_service: LLMService,
        max_iterations: int = 5,
        require_approval_per_iteration: bool = True
    ):
        """
        Initialize TPAO loop.
        
        Args:
            rag_pipeline: RAG pipeline for context retrieval
            tool_registry: Registry of available tools
            llm_service: LLM service for reasoning
            max_iterations: Maximum number of TPAO iterations
            require_approval_per_iteration: If True, requires human approval after each iteration (SAFER)
        """
        self.rag_pipeline = rag_pipeline
        self.tool_registry = tool_registry
        self.llm_service = llm_service
        self.max_iterations = max_iterations
        self.require_approval_per_iteration = require_approval_per_iteration
        self.logger = logging.getLogger(__name__)
    
    def execute(
        self,
        issue: Issue,
        initial_context: Optional[str] = None
    ) -> Resolution:
        """
        Execute TPAO loop for a novel issue.
        
        Args:
            issue: Customer issue to resolve
            initial_context: Optional initial context from orchestrator
        
        Returns:
            Resolution with proposed actions
        """
        self.logger.info(f"Starting TPAO loop for issue {issue.issue_id}")
        
        # Initialize resolution
        resolution = Resolution(
            resolution_id=f"res_tpao_{issue.issue_id}_{int(datetime.now().timestamp())}",
            issue_id=issue.issue_id,
            status=ResolutionStatus.IN_PROGRESS,
            novel_task=True,
            skill_matched=False
        )
        
        # Initialize state
        state = TPAOState(
            issue=issue,
            resolution=resolution,
            current_phase=TPAOPhase.THINK,
            iteration=0,
            max_iterations=self.max_iterations,
            context_history=[initial_context] if initial_context else [],
            action_history=[],
            observations=[],
            is_resolved=False,
            needs_approval=False
        )
        
        # Execute loop
        while not state.is_resolved and state.iteration < state.max_iterations:
            state.iteration += 1
            self.logger.info(
                f"TPAO iteration {state.iteration}/{state.max_iterations}, "
                f"phase: {state.current_phase}"
            )
            
            try:
                if state.current_phase == TPAOPhase.THINK:
                    state = self._think_phase(state)
                elif state.current_phase == TPAOPhase.PLAN:
                    state = self._plan_phase(state)
                elif state.current_phase == TPAOPhase.ACT:
                    state = self._act_phase(state)
                elif state.current_phase == TPAOPhase.OBSERVE:
                    state = self._observe_phase(state)
                
                # Check if we need to exit for approval (but only after completing all phases in first iteration)
                # Don't break during Think phase - let it complete the full TPAO cycle first
                if state.needs_approval and state.current_phase == TPAOPhase.OBSERVE:
                    self.logger.info("TPAO loop pausing for human approval after completing cycle")
                    break
                    
            except Exception as e:
                self.logger.error(f"TPAO loop error in {state.current_phase}: {e}")
                state.error = str(e)
                state.resolution.mark_failed(f"TPAO loop failed: {e}")
                break
        
        # Finalize resolution
        if state.iteration >= state.max_iterations and not state.is_resolved:
            self.logger.warning(f"TPAO loop reached max iterations without resolution")
            state.resolution.mark_escalated("Max iterations reached, requires human review")
        
        return state.resolution
    
    def _think_phase(self, state: TPAOState) -> TPAOState:
        """
        Think Phase: Analyze issue and identify information needs using LLM.
        
        This phase:
        1. Uses LLM to analyze the customer issue
        2. Identifies what information is needed
        3. Queries RAG for relevant context
        4. Updates context history
        
        Args:
            state: Current TPAO state
        
        Returns:
            Updated state
        """
        self.logger.info(f"[Think] Analyzing issue: {state.issue.body[:100]}...")
        
        step_start = datetime.now()
        
        # Build analysis prompt for LLM
        analysis_prompt = f"""You are a hotel customer service AI assistant analyzing a customer issue.

Customer Issue:
- Channel: {state.issue.channel}
- Subject: {state.issue.subject or 'N/A'}
- Message: {state.issue.body}
- Booking ID: {state.issue.booking_id or 'Not provided'}
- Guest Email: {state.issue.guest_email or 'Not provided'}

Previous Observations: {' | '.join(state.observations[-2:]) if state.observations else 'None'}

Analyze this issue and identify:
1. What is the core problem or request?
2. What information do we need to retrieve to resolve this?
3. What hotel policies might be relevant?

Provide your analysis in 2-3 sentences."""

        try:
            # Get LLM analysis
            llm_response = self.llm_service.generate_simple(analysis_prompt)
            analysis = llm_response.content
            
            self.logger.info(f"[Think] LLM Analysis: {analysis[:200]}...")
            
            # Build RAG query based on LLM analysis and issue
            query_parts = [state.issue.body, analysis]
            if state.observations:
                query_parts.append(f"Previous observations: {' '.join(state.observations[-2:])}")
            
            query = " ".join(query_parts)
            
            # Retrieve context from RAG
            # Simple filter: just get policy documents (all our indexed docs are policies)
            metadata_filters = {
                "doc_type": "policy"
            }
            
            reranked_results, rag_metrics = self.rag_pipeline.query_with_reranking(
                query=query,
                metadata_filters=metadata_filters,
                top_k=5
            )
            
            # GUARDRAIL: Check if we have sufficient context
            if len(reranked_results) == 0:
                self.logger.warning(f"[Think] No relevant policies found for issue {state.issue.issue_id} - using LLM general knowledge")
                
                # Use LLM's general knowledge but add strong warning
fallback_prompt = f"""️ WARNING: No company policies found in knowledge base for this issue.
Providing response based on general knowledge only. This response MUST be reviewed by a human before taking action.

Issue: {state.issue.subject}
Details: {state.issue.body}

Based on general hospitality industry best practices (NOT company-specific policies), provide:
1. A general understanding of the issue
2. Common approaches to handle such requests
3. Important considerations

Remember: This is NOT based on company policy and requires human review."""

                fallback_response = self.llm_service.generate_simple(fallback_prompt)
fallback_context = f"️ NO POLICY MATCH - GENERAL KNOWLEDGE ONLY:\n\n{fallback_response.content}"
                
                # Store as context with warning
                state.context_history.append(fallback_context)
                
                # Record step with warning
                step = ResolutionStep(
                    step_number=len(state.resolution.steps) + 1,
                    step_type="think",
description="️ No relevant policies found - using LLM general knowledge (REQUIRES HUMAN REVIEW)",
                    success=True,  # Continue but with warning
                    input_data={"query": query, "analysis": analysis, "top_k": 5},
                    output_data={
                        "num_results": 0,
                        "warning": "NO_POLICY_MATCH_USING_GENERAL_KNOWLEDGE",
                        "fallback_used": True,
                        "llm_tokens": llm_response.tokens_used + fallback_response.tokens_used,
                        "citations": [{
                            "source_id": 0,
                            "source_name": "LLM General Knowledge (NOT company policy)",
                            "doc_type": "fallback",
                            "relevance_score": 0.0,
                            "content_preview": fallback_response.content[:200]
                        }]
                    },
                    duration_ms=(datetime.now() - step_start).total_seconds() * 1000
                )
                state.resolution.add_step(step)
                
                # Mark as requiring human review
                state.resolution.metadata["no_policy_match"] = True
                state.resolution.metadata["using_general_knowledge"] = True
                state.needs_approval = True  # Force human review
                
                # DON'T return - let it continue to normal context processing
                # The fallback_context is already added to state.context_history above
                
            else:
                # Normal path: we have policy results
                # GUARDRAIL: Check if confidence is too low (average score < 0.3)
                avg_score = sum(r.rerank_score for r in reranked_results) / len(reranked_results) if reranked_results else 0
                if avg_score < 0.3:
                    self.logger.warning(f"[Think] Low confidence retrieval (avg score: {avg_score:.3f}) for issue {state.issue.issue_id}")
                    # Add warning but continue with caution
                    state.resolution.metadata["low_confidence_warning"] = True
                    state.resolution.metadata["avg_retrieval_score"] = avg_score
                
                # Extract and store context with citations
                context_parts = []
                citations = []
                
                for i, r in enumerate(reranked_results[:3], 1):
                    # Extract source information from metadata
                    source_name = r.retrieval_result.metadata.get('source', 'Unknown Source')
                    doc_type = r.retrieval_result.metadata.get('doc_type', 'document')
                    score = r.rerank_score
                    
                    # Format citation
                    citation = {
                        "source_id": i,
                        "source_name": source_name,
                        "doc_type": doc_type,
                        "relevance_score": round(score, 3),
                        "content_preview": r.retrieval_result.content[:200]
                    }
                    citations.append(citation)
                    
                    # Format context with citation
                    context_parts.append(
                        f"[Source {i}: {source_name}] (relevance: {score:.2f})\n"
                        f"{r.retrieval_result.content[:400]}..."
                    )
                
                context = "\n\n".join(context_parts)
                state.context_history.append(context)
                
                # Record step with citations
                step = ResolutionStep(
                    step_number=len(state.resolution.steps) + 1,
                    step_type="think",
                    description=f"LLM analyzed issue and retrieved {len(reranked_results)} relevant context pieces from policy knowledge base",
                    success=True,
                    input_data={"query": query, "analysis": analysis, "top_k": 5},
                    output_data={
                        "num_results": len(reranked_results),
                        "avg_score": sum(r.rerank_score for r in reranked_results) / len(reranked_results) if reranked_results else 0,
                        "llm_tokens": llm_response.tokens_used,
                        "citations": citations  # Add citations to output
                    },
                    duration_ms=(datetime.now() - step_start).total_seconds() * 1000
                )
                state.resolution.add_step(step)
            
            # Move to plan phase
            state.current_phase = TPAOPhase.PLAN
            
        except Exception as e:
            self.logger.error(f"[Think] Analysis failed: {e}")
            step = ResolutionStep(
                step_number=len(state.resolution.steps) + 1,
                step_type="think",
                description="LLM analysis or RAG retrieval failed",
                success=False,
                error_message=str(e),
                duration_ms=(datetime.now() - step_start).total_seconds() * 1000
            )
            state.resolution.add_step(step)
            state.error = str(e)
        
        return state
    
    def _plan_phase(self, state: TPAOState) -> TPAOState:
        """
        Plan Phase: Draft resolution steps using RAG context and LLM reasoning.
        
        This phase:
        1. Uses LLM to analyze retrieved context
        2. Generates structured resolution plan
        3. Identifies appropriate tools
        4. Determines approval requirements
        
        Args:
            state: Current TPAO state
        
        Returns:
            Updated state
        """
        self.logger.info("[Plan] Drafting resolution steps with LLM")
        
        step_start = datetime.now()
        
        # Get latest context
        latest_context = state.context_history[-1] if state.context_history else ""
        
        # Get available tools
        available_tools = list(self.tool_registry.list_tools())
        tools_description = "\n".join([
            f"- {tool}: {self.tool_registry.get_tool(tool).__doc__ or 'No description'}"
            for tool in available_tools
        ])
        
        # Build planning prompt
        planning_prompt = f"""You are a hotel customer service AI assistant creating a resolution plan.

Customer Issue:
- Channel: {state.issue.channel}
- Subject: {state.issue.subject or 'N/A'}
- Message: {state.issue.body}
- Booking ID: {state.issue.booking_id or 'Not provided'}

Retrieved Context:
{latest_context}

Available Tools:
{tools_description}

Create a step-by-step resolution plan. For each step, specify:
1. A clear description of what to do
2. Which tool to use (if applicable)
3. What parameters the tool needs
4. Whether this step requires human approval (true for financial actions, booking changes, or sensitive operations)

Respond with a JSON array of steps in this format:
[
  {{
    "step_id": "step_1",
    "description": "Look up booking details",
    "tool_name": "lookup_booking",
    "parameters": {{"booking_id": "BK12345"}},
    "requires_approval": false,
    "reasoning": "Need booking info to proceed"
  }},
  {{
    "step_id": "step_2",
    "description": "Process refund",
    "tool_name": "process_refund",
    "parameters": {{"booking_id": "BK12345", "amount": "full"}},
    "requires_approval": true,
    "reasoning": "Financial action requires approval"
  }}
]

Generate 2-4 steps. Be specific and actionable."""

        try:
            # Get LLM plan
            plan_steps = self.llm_service.generate_json_simple(planning_prompt)
            
            # Validate the plan
            if not isinstance(plan_steps, list):
                self.logger.warning(f"[Plan] LLM returned non-list response, using fallback")
                plan_steps = self._generate_fallback_plan(state.issue)
            
            self.logger.info(f"[Plan] Generated {len(plan_steps)} steps")
            
            # Record step
            step = ResolutionStep(
                step_number=len(state.resolution.steps) + 1,
                step_type="plan",
                description=f"LLM generated plan with {len(plan_steps)} actions",
                success=True,
                input_data={"context_length": len(latest_context), "available_tools": available_tools},
                output_data={
                    "num_actions": len(plan_steps),
                    "actions": plan_steps
                },
                duration_ms=(datetime.now() - step_start).total_seconds() * 1000
            )
            state.resolution.add_step(step)
            
            # Store actions for act phase
            state.action_history.extend(plan_steps)
            
            # Move to act phase
            state.current_phase = TPAOPhase.ACT
            
        except Exception as e:
            self.logger.error(f"[Plan] Planning failed: {e}")
            step = ResolutionStep(
                step_number=len(state.resolution.steps) + 1,
                step_type="plan",
                description="LLM planning failed",
                success=False,
                error_message=str(e),
                duration_ms=(datetime.now() - step_start).total_seconds() * 1000
            )
            state.resolution.add_step(step)
            state.error = str(e)
        
        return state
    
    def _act_phase(self, state: TPAOState) -> TPAOState:
        """
        Act Phase: Convert LLM plan to ResolutionSteps and prepare for approval.
        
        This phase:
        1. Reviews planned actions from LLM
        2. Converts them to ResolutionStep objects
        3. Identifies which actions need approval
        4. Pauses for human review if needed
        
        Args:
            state: Current TPAO state
        
        Returns:
            Updated state
        """
        self.logger.info("[Act] Converting plan to resolution steps")
        
        step_start = datetime.now()
        
        # Track how many actions we've already processed
        # This prevents duplicate steps across iterations
        if not hasattr(state, '_actions_processed_count'):
            state._actions_processed_count = 0
        
        # Get only NEW actions from this iteration (not already processed)
        all_actions = state.action_history
        current_actions = all_actions[state._actions_processed_count:]
        
        # Update the count for next iteration
        state._actions_processed_count = len(all_actions)
        
        self.logger.info(f"[Act] Processing {len(current_actions)} new actions (total in history: {len(all_actions)})")
        
        # Convert actions to ResolutionSteps
        for action in current_actions:
            action_step = ResolutionStep(
                step_number=len(state.resolution.steps) + 1,
                step_type="action",
                description=action.get("description", "Planned action"),
                tool_used=action.get("tool_name"),
                input_data={
                    "parameters": action.get("parameters", {}),
                    "reasoning": action.get("reasoning", "")
                },
                output_data={"requires_approval": action.get("requires_approval", False)},
                success=True,  # Planned, not yet executed
                duration_ms=0
            )
            state.resolution.add_step(action_step)
        
        # Only process if there are new actions
        if len(current_actions) > 0:
            # Determine if any actions need approval
            approval_needed = any(
                action.get("requires_approval", False)
                for action in current_actions
            )
            
            self.logger.info(f"[Act] {len(current_actions)} actions prepared, approval_needed={approval_needed}")
            state.needs_approval = True
            state.resolution.status = ResolutionStatus.PENDING_APPROVAL
            
            # Store actions in resolution metadata
            state.resolution.metadata["proposed_actions"] = current_actions
            state.resolution.metadata["approval_reason"] = "Novel task requires human review before execution"
            
            # Record act phase step (only when there are actions)
            step = ResolutionStep(
                step_number=len(state.resolution.steps) + 1,
                step_type="act",
                description=f"Prepared {len(current_actions)} actions for approval",
                success=True,
                input_data={"num_actions": len(current_actions)},
                output_data={
                    "requires_approval": approval_needed,
                    "action_count": len(current_actions)
                },
                duration_ms=(datetime.now() - step_start).total_seconds() * 1000
            )
            state.resolution.add_step(step)
        else:
            # No new actions in this iteration, move to observe
            self.logger.info("[Act] No new actions in this iteration, moving to observe")
            state.current_phase = TPAOPhase.OBSERVE
        
        return state
    
    def _observe_phase(self, state: TPAOState) -> TPAOState:
        """
        Observe Phase: Record results and update context.
        
        This phase:
        1. Records action results
        2. Updates observations
        3. Determines if issue is resolved
        4. Decides next iteration or completion
        
        Args:
            state: Current TPAO state
        
        Returns:
            Updated state
        """
        self.logger.info("[Observe] Recording results")
        
        step_start = datetime.now()
        
        # Create observation
        observation = f"Iteration {state.iteration}: Generated {len(state.action_history)} total actions, {len([a for a in state.action_history if a.get('requires_approval')])} require approval"
        state.observations.append(observation)
        
        # Check if we should continue or resolve
        if state.needs_approval:
            state.is_resolved = True
            self.logger.info("[Observe] Issue ready for approval - TPAO loop complete")
        elif state.iteration >= state.max_iterations:
            state.is_resolved = True
            state.needs_approval = True  # Force approval even if no explicit approval actions
            state.resolution.status = ResolutionStatus.PENDING_APPROVAL
            self.logger.info("[Observe] Max iterations reached - sending for approval")
        elif self.require_approval_per_iteration:
            # SAFETY: Require human approval after EACH iteration before continuing
            state.needs_approval = True
            state.is_resolved = True
            state.resolution.status = ResolutionStatus.PENDING_APPROVAL
            state.resolution.metadata["iteration_approval_required"] = True
            state.resolution.metadata["iteration_number"] = state.iteration
            self.logger.info(f"[Observe] Iteration {state.iteration} complete - requiring human approval before continuing")
        else:
            # Continue to next iteration (only if approval per iteration is disabled)
            state.current_phase = TPAOPhase.THINK
            self.logger.info("[Observe] Continuing to next iteration without approval")
        
        # Record step
        step = ResolutionStep(
            step_number=len(state.resolution.steps) + 1,
            step_type="observe",
            description=observation,
            success=True,
            input_data={"iteration": state.iteration},
            output_data={
                "is_resolved": state.is_resolved,
                "needs_approval": state.needs_approval,
                "total_actions": len(state.action_history)
            },
            duration_ms=(datetime.now() - step_start).total_seconds() * 1000
        )
        state.resolution.add_step(step)
        
        return state
    
    def _generate_fallback_plan(self, issue: Issue) -> List[Dict[str, Any]]:
        """
        Generate a simple fallback plan when LLM fails.
        
        Args:
            issue: Customer issue
        
        Returns:
            List of action dictionaries
        """
        actions = []
        
        # Always start with looking up booking if we have a booking_id
        if issue.booking_id:
            actions.append({
                "step_id": "lookup_booking",
                "description": "Look up booking details",
                "tool_name": "lookup_booking",
                "parameters": {"booking_id": issue.booking_id},
                "requires_approval": False,
                "reasoning": "Need booking context"
            })
        
        # Add a generic response action
        actions.append({
            "step_id": "send_response",
            "description": "Send response to customer",
            "tool_name": "send_email",
            "parameters": {
                "to": issue.guest_email or "customer",
                "subject": f"Re: {issue.subject or 'Your request'}"
            },
            "requires_approval": True,
            "reasoning": "Customer communication requires approval"
        })
        
        return actions
