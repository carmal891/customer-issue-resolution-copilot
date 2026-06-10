"""
Orchestrator Agent using LangGraph

This is the central coordinator that:
1. Classifies customer issues
2. Retrieves relevant context via RAG
3. Matches issues to existing skills
4. Routes to skill path or novel task path
5. Proposes actions for human approval

LangGraph Concepts Used:
- StateGraph: Defines the workflow
- Nodes: Individual processing steps
- Conditional Edges: Routing logic
- State: Data flowing through the graph
"""

from typing import TypedDict, List, Optional, Dict, Any, Annotated
from dataclasses import dataclass
import logging
from datetime import datetime

# LangGraph imports
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# Our domain models
from src.domain.models.issue import Issue
from src.domain.models.resolution import Resolution
from src.domain.models.context import RetrievedContext, RAGContext

# Our components
from src.application.rag.rag_pipeline import RAGPipeline
from src.application.skills.skill_matcher import SkillMatcher, SkillMatch, MatchConfidence
from src.application.tools.base import ToolRegistry

logger = logging.getLogger(__name__)


# ============================================================================
# STATE DEFINITION
# ============================================================================
# This is the data structure that flows through our graph.
# Each node reads from and writes to this state.

class OrchestratorState(TypedDict):
    """
    State that flows through the orchestrator graph.

    LangGraph Concept: State is a TypedDict that gets passed between nodes.
    Each node can read any field and update any field.
    """
    # ===== INPUT =====
    issue: Issue  # The customer issue to resolve

    # ===== CLASSIFICATION =====
    intent: Optional[str]  # What the customer wants (refund, upgrade, etc.)
    category: Optional[str]  # Issue category (billing, booking, etc.)
    priority: Optional[str]  # Priority level (low, medium, high, urgent)

    # ===== RAG CONTEXT =====
    retrieved_docs: List[str]  # Raw retrieved documents
    context_summary: Optional[str]  # Summarized context for LLM
    retrieval_confidence: Optional[float]  # Confidence in retrieved context

    # ===== SKILL MATCHING =====
    skill_matches: List[SkillMatch]  # All matching skills
    best_match: Optional[SkillMatch]  # Best matching skill

    # ===== SKILL PROPOSAL APPROVAL =====
    skill_proposal_approved: Optional[bool]  # Whether human approved the skill match
    skill_rejection_reason: Optional[str]  # Why skill was rejected

    # ===== ROUTING =====
    route: Optional[str]  # "skill_path" or "novel_path"

    # ===== EXECUTION PLAN =====
    proposed_actions: List[Dict[str, Any]]  # Actions to execute
    requires_approval: bool  # Whether human approval is needed
    approval_reason: Optional[str]  # Why approval is needed

    # ===== RESULTS =====
    resolution: Optional[Resolution]  # Final resolution

    # ===== ERROR HANDLING =====
    error: Optional[str]  # Error message if something fails
    retry_count: int  # Number of retries attempted

    # ===== METADATA =====
    trace_id: str  # Unique trace ID for logging
    start_time: str  # When processing started
    messages: List[Dict[str, str]]  # Conversation history


# ============================================================================
# ORCHESTRATOR AGENT CLASS
# ============================================================================

class OrchestratorAgent:
    """
    Orchestrator Agent using LangGraph.

    This agent coordinates the entire issue resolution workflow:
    1. Classify issue → 2. Retrieve context → 3. Match skill → 4. Route → 5. Execute

    LangGraph Architecture:
    - Nodes: Individual processing functions
    - Edges: Connections between nodes
    - Conditional Edges: Routing based on state
    - Graph: Compiled workflow
    """

    def __init__(
        self,
        rag_pipeline: RAGPipeline,
        skill_matcher: SkillMatcher,
        tool_registry: ToolRegistry,
        llm: Any  # LangChain LLM instance
    ):
        """
        Initialize orchestrator with required components.

        Args:
            rag_pipeline: RAG system for context retrieval
            skill_matcher: Skill matching system
            tool_registry: Registry of available tools
            llm: LangChain LLM for classification and planning
        """
        self.rag_pipeline = rag_pipeline
        self.skill_matcher = skill_matcher
        self.tool_registry = tool_registry
        self.llm = llm

        # Build the LangGraph workflow
        self.graph = self._build_graph()

        logger.info("OrchestratorAgent initialized with LangGraph")

    # ========================================================================
    # GRAPH CONSTRUCTION
    # ========================================================================

    def _build_graph(self) -> StateGraph:
        """
        Build the LangGraph workflow.

        LangGraph Concept: We define nodes, edges, and conditional routing.
        The graph represents our entire workflow as a state machine.

        Updated Workflow (Human-Rejection-Based Routing):
        START → classify → retrieve → match → route_decision
          ↓
          Has match?
            Yes → skill_proposal (human approval) → approval_decision
              ↓
              Approved?
                Yes → skill_path → prepare_approval → END
                No → novel_path (TPAO) → prepare_approval → END
            No → novel_path (TPAO) → prepare_approval → END
        """
        # Create the graph with our state type
        workflow = StateGraph(OrchestratorState)

        # ===== ADD NODES =====
        # Each node is a function that processes state
        workflow.add_node("classify_issue", self._classify_issue_node)
        workflow.add_node("retrieve_context", self._retrieve_context_node)
        workflow.add_node("match_skill", self._match_skill_node)
        workflow.add_node("skill_proposal", self._skill_proposal_node)  # NEW: Human approval node
        workflow.add_node("skill_path", self._skill_path_node)
        workflow.add_node("novel_path", self._novel_path_node)
        workflow.add_node("prepare_approval", self._prepare_approval_node)
        workflow.add_node("handle_error", self._handle_error_node)

        # ===== ADD NORMAL EDGES =====
        # These edges always go to the next node
        workflow.add_edge("classify_issue", "retrieve_context")
        workflow.add_edge("retrieve_context", "match_skill")
        workflow.add_edge("skill_path", "prepare_approval")
        workflow.add_edge("novel_path", "prepare_approval")
        workflow.add_edge("prepare_approval", END)
        workflow.add_edge("handle_error", END)

        # ===== ADD CONDITIONAL EDGES =====
        # These edges route based on state

        # After matching, route to skill proposal or novel path (no confidence check)
        workflow.add_conditional_edges(
            "match_skill",
            self._route_decision,  # Returns "skill_proposal" or "novel_path"
            {
                "skill_proposal": "skill_proposal",  # NEW: Route to human approval
                "novel_path": "novel_path",
                "error": "handle_error"
            }
        )

        # NEW: After skill proposal, route based on human decision
        workflow.add_conditional_edges(
            "skill_proposal",
            self._skill_approval_decision,  # Returns "skill_path" or "novel_path"
            {
                "skill_path": "skill_path",  # Human approved → use skill
                "novel_path": "novel_path"   # Human rejected → TPAO
            }
        )

        # ===== SET ENTRY POINT =====
        workflow.set_entry_point("classify_issue")

        # ===== COMPILE GRAPH =====
        # This creates the executable workflow
        compiled_graph = workflow.compile()

        logger.info("LangGraph workflow compiled successfully with human-rejection-based routing")
        return compiled_graph

    # ========================================================================
    # NODE IMPLEMENTATIONS
    # ========================================================================
    # Each node is a function that:
    # 1. Takes current state as input
    # 2. Performs some operation
    # 3. Returns updated state

    def _classify_issue_node(self, state: OrchestratorState) -> OrchestratorState:
        """
        Node: Classify the customer issue.

        LangGraph Concept: This is a NODE - a processing step in the graph.
        It reads from state, does work, and updates state.

        Inputs from state:
        - issue: The customer issue

        Outputs to state:
        - intent: What customer wants
        - category: Issue category
        - priority: Priority level
        """
        try:
            logger.info(f"[{state['trace_id']}] Classifying issue: {state['issue'].issue_id}")

            issue = state["issue"]

            # Build classification prompt
            classification_prompt = f"""
            Classify this customer issue:
            
            Title: {issue.title}
            Description: {issue.description}
            Channel: {issue.channel}
            
            Provide:
            1. Intent (what they want): refund, upgrade, complaint, question, etc.
            2. Category: billing, booking, amenity, access, technical
            3. Priority: low, medium, high, urgent
            
            Format as JSON.
            """

            # Use LLM to classify
            messages = [
                SystemMessage(content="You are a customer service issue classifier."),
                HumanMessage(content=classification_prompt)
            ]

            response = self.llm.invoke(messages)

            # Parse response (simplified - in production use structured output)
            # For now, use simple heuristics
            description_lower = issue.description.lower()

            # Determine intent
            if any(word in description_lower for word in ["refund", "money back", "cancel"]):
                intent = "refund"
            elif any(word in description_lower for word in ["upgrade", "better room"]):
                intent = "upgrade"
            elif any(word in description_lower for word in ["late checkout", "extend"]):
                intent = "late_checkout"
            elif any(word in description_lower for word in ["complaint", "unhappy", "disappointed"]):
                intent = "complaint"
            else:
                intent = "general_inquiry"

            # Determine category
            if any(word in description_lower for word in ["bill", "charge", "payment"]):
                category = "billing"
            elif any(word in description_lower for word in ["room", "booking", "reservation"]):
                category = "booking"
            elif any(word in description_lower for word in ["amenity", "service", "facility"]):
                category = "amenity"
            else:
                category = "general"

            # Determine priority
            if any(word in description_lower for word in ["urgent", "asap", "immediately"]):
                priority = "urgent"
            elif any(word in description_lower for word in ["important", "soon"]):
                priority = "high"
            else:
                priority = "medium"

            # Update state
            state["intent"] = intent
            state["category"] = category
            state["priority"] = priority

            # Add to messages
            state["messages"].append({
                "role": "system",
                "content": f"Classified as: {intent} ({category}, {priority})"
            })

            logger.info(f"[{state['trace_id']}] Classification: {intent}, {category}, {priority}")

            return state

        except Exception as e:
            logger.error(f"[{state['trace_id']}] Classification failed: {e}")
            state["error"] = f"Classification error: {e}"
            return state

    def _retrieve_context_node(self, state: OrchestratorState) -> OrchestratorState:
        """
        Node: Retrieve relevant context using RAG.

        Inputs from state:
        - issue: Customer issue
        - intent: Classified intent
        - category: Classified category

        Outputs to state:
        - retrieved_docs: Retrieved documents
        - context_summary: Summarized context
        - retrieval_confidence: Confidence score
        """
        try:
            logger.info(f"[{state['trace_id']}] Retrieving context via RAG")

            issue = state["issue"]

            # Build enhanced query using classification
            query = f"{issue.description}"
            if state.get("intent"):
                query += f" Intent: {state['intent']}"
            if state.get("category"):
                query += f" Category: {state['category']}"

            # Retrieve using RAG pipeline with reranking
            reranked_results, rag_metrics = self.rag_pipeline.query_with_reranking(
                query=query,
                metadata_filters={"domain": state.get("category")} if state.get("category") else None,
                top_k=5
            )

            # Extract documents from reranked results
            retrieved_docs = [r.retrieval_result.content for r in reranked_results]

            # Calculate average confidence from rerank scores
            if reranked_results:
                avg_confidence = sum(r.rerank_score for r in reranked_results) / len(reranked_results)
            else:
                avg_confidence = 0.0

            # Assemble context summary from retrieved content
            context_summary = "\n\n".join([
                f"[Source {i+1}] {r.retrieval_result.content[:500]}..."
                for i, r in enumerate(reranked_results[:3])
            ])

            # Update state
            state["retrieved_docs"] = retrieved_docs
            state["context_summary"] = context_summary
            state["retrieval_confidence"] = avg_confidence

            # Add to messages
            state["messages"].append({
                "role": "system",
                "content": f"Retrieved {len(retrieved_docs)} relevant documents (confidence: {avg_confidence:.2f})"
            })

            logger.info(f"[{state['trace_id']}] Retrieved {len(retrieved_docs)} docs, confidence: {avg_confidence:.2f}")

            return state

        except Exception as e:
            logger.error(f"[{state['trace_id']}] Context retrieval failed: {e}")
            state["error"] = f"Retrieval error: {e}"
            return state

    def _match_skill_node(self, state: OrchestratorState) -> OrchestratorState:
        """
        Node: Match issue to existing skills.

        Inputs from state:
        - issue: Customer issue

        Outputs to state:
        - skill_matches: All matching skills
        - best_match: Best matching skill
        """
        try:
            logger.info(f"[{state['trace_id']}] Matching skills")

            issue = state["issue"]

            # Match skills
            matches = self.skill_matcher.match_skill(
                issue=issue,
                top_k=3,
                min_confidence=MatchConfidence.LOW
            )

            # Update state
            state["skill_matches"] = matches
            state["best_match"] = matches[0] if matches else None

            # Add to messages
            if matches:
                match_info = f"Found {len(matches)} matching skills. Best: {matches[0].skill.name} (confidence: {matches[0].confidence})"
            else:
                match_info = "No matching skills found"

            state["messages"].append({
                "role": "system",
                "content": match_info
            })

            logger.info(f"[{state['trace_id']}] {match_info}")

            return state

        except Exception as e:
            logger.error(f"[{state['trace_id']}] Skill matching failed: {e}")
            state["error"] = f"Skill matching error: {e}"
            return state

    def _skill_proposal_node(self, state: OrchestratorState) -> OrchestratorState:
        """
        Node: Present skill proposal to human for approval.

        This is a placeholder for human interaction.
        In a real system, this would:
        1. Display the matched skill to the human
        2. Show the confidence score and reasoning
        3. Wait for human to approve or reject
        4. Capture rejection reason if rejected

        For now, we'll simulate this with a simple check.

        Inputs from state:
        - best_match: Matched skill

        Outputs to state:
        - skill_proposal_approved: True/False
        - skill_rejection_reason: Reason if rejected
        """
        try:
            logger.info(f"[{state['trace_id']}] Presenting skill proposal to human")

            best_match = state.get("best_match")
            if not best_match or not best_match.skill:
                logger.error(f"[{state['trace_id']}] No skill found in best_match")
                state["error"] = "No skill available for proposal"
                return state

            skill = best_match.skill

            # In a real system, this would be an interactive prompt
            # For now, we'll auto-approve for demonstration
            # In production, this would call an approval interface

            # Simulate human approval (in real system, this comes from UI/CLI)
            # For testing, we can set this based on confidence or other criteria
            # For now, let's auto-approve to maintain current behavior
            state["skill_proposal_approved"] = True
            state["skill_rejection_reason"] = None

            # Add to messages
            state["messages"].append({
                "role": "system",
                "content": f"Skill proposal: {skill.name} (score: {best_match.score:.3f}, confidence: {best_match.confidence})"
            })

            state["messages"].append({
                "role": "human",
                "content": "Skill approved" if state["skill_proposal_approved"] else f"Skill rejected: {state['skill_rejection_reason']}"
            })

            logger.info(f"[{state['trace_id']}] Skill proposal: {'approved' if state['skill_proposal_approved'] else 'rejected'}")

            return state

        except Exception as e:
            logger.error(f"[{state['trace_id']}] Skill proposal failed: {e}")
            state["error"] = f"Skill proposal error: {e}"
            return state

    def _skill_path_node(self, state: OrchestratorState) -> OrchestratorState:
        """
        Node: Execute skill path (existing skill found).

        Inputs from state:
        - best_match: Matched skill
        - context_summary: RAG context

        Outputs to state:
        - proposed_actions: Actions from skill
        - requires_approval: Whether approval needed
        """
        try:
            logger.info(f"[{state['trace_id']}] Executing skill path")

            best_match = state.get("best_match")
            if not best_match or not best_match.skill:
                logger.error(f"[{state['trace_id']}] No skill found in best_match")
                state["error"] = "No skill available for execution"
                return state

            skill = best_match.skill

            # Extract actions from skill steps
            proposed_actions = []
            requires_approval = False

            for step in skill.steps:
                action = {
                    "step_id": step.get("step_id"),
                    "type": step.get("step_type"),
                    "description": step.get("description"),
                    "tool_name": step.get("tool_name"),
                    "parameters": step.get("parameters", {}),
                    "requires_approval": step.get("requires_approval", False)
                }
                proposed_actions.append(action)

                if action["requires_approval"]:
                    requires_approval = True

            # Update state
            state["proposed_actions"] = proposed_actions
            state["requires_approval"] = requires_approval
            state["route"] = "skill_path"

            if requires_approval:
                state["approval_reason"] = f"Skill '{skill.name}' requires approval for sensitive actions"

            # Add to messages
            state["messages"].append({
                "role": "assistant",
                "content": f"Using skill: {skill.name}. Proposed {len(proposed_actions)} actions."
            })

            logger.info(f"[{state['trace_id']}] Skill path: {len(proposed_actions)} actions, approval: {requires_approval}")

            return state

        except Exception as e:
            logger.error(f"[{state['trace_id']}] Skill path failed: {e}")
            state["error"] = f"Skill path error: {e}"
            return state

    def _novel_path_node(self, state: OrchestratorState) -> OrchestratorState:
        """
        Node: Execute novel path (no skill found, need ReAct loop).

        This is a placeholder - the full ReAct loop will be implemented separately.

        Inputs from state:
        - issue: Customer issue
        - context_summary: RAG context

        Outputs to state:
        - proposed_actions: Generated actions
        - requires_approval: Always true for novel tasks
        """
        try:
            logger.info(f"[{state['trace_id']}] Executing novel path (ReAct loop)")

            # For now, create a simple plan
            # In full implementation, this will call the ReAct/TPAO loop

            issue = state["issue"]
            context = state.get("context_summary", "")

            # Generate plan using LLM
            planning_prompt = f"""
            Create a resolution plan for this customer issue:
            
            Issue: {issue.description}
            Intent: {state.get('intent')}
            Category: {state.get('category')}
            
            Context from knowledge base:
            {context}
            
            Provide a step-by-step plan with specific actions.
            """

            messages = [
                SystemMessage(content="You are a customer service resolution planner."),
                HumanMessage(content=planning_prompt)
            ]

            response = self.llm.invoke(messages)

            # Create proposed actions (simplified)
            proposed_actions = [
                {
                    "step_id": "investigate",
                    "type": "tool_call",
                    "description": "Investigate issue details",
                    "tool_name": "lookup_booking",
                    "parameters": {"booking_id": issue.metadata.get("booking_id", "unknown")},
                    "requires_approval": False
                },
                {
                    "step_id": "resolve",
                    "type": "tool_call",
                    "description": "Execute resolution action",
                    "tool_name": "send_email",
                    "parameters": {"to": "customer", "subject": "Resolution"},
                    "requires_approval": True
                }
            ]

            # Update state
            state["proposed_actions"] = proposed_actions
            state["requires_approval"] = True  # Novel tasks always need approval
            state["route"] = "novel_path"
            state["approval_reason"] = "Novel task requires human review before execution"

            # Add to messages
            state["messages"].append({
                "role": "assistant",
                "content": f"Generated novel resolution plan with {len(proposed_actions)} actions."
            })

            logger.info(f"[{state['trace_id']}] Novel path: {len(proposed_actions)} actions generated")

            return state

        except Exception as e:
            logger.error(f"[{state['trace_id']}] Novel path failed: {e}")
            state["error"] = f"Novel path error: {e}"
            return state

    def _prepare_approval_node(self, state: OrchestratorState) -> :
        """
        Node: Prepare approval request.

        Inputs from state:
        - proposed_actions: Actions to execute
        - requires_approval: Whether approval needed

        Outputs to state:
        - resolution: Prepared resolution (pending approval)
        """
        try:
            logger.info(f"[{state['trace_id']}] Preparing approval request")

            # Create resolution object with correct fields from Resolution model
            from src.domain.models.resolution import ResolutionStatus

            # Safely extract skill_id if best_match exists
            best_match = state.get("best_match")
            skill_id = None
            if best_match and hasattr(best_match, 'skill') and best_match.skill:
                skill_id = best_match.skill.skill_id

            resolution = Resolution(
                resolution_id=f"res_{state['trace_id']}",
                issue_id=state["issue"].issue_id,
                status=ResolutionStatus.PENDING_APPROVAL if state["requires_approval"] else ResolutionStatus.IN_PROGRESS,
                skill_used=skill_id,
                skill_matched=bool(best_match),
                novel_task=state.get("route") == "novel_path",
                metadata={
                    "proposed_actions": state["proposed_actions"],
                    "requires_approval": state["requires_approval"],
                    "approval_reason": state.get("approval_reason"),
                    "context_used": state.get("context_summary"),
                    "route_taken": state.get("route"),
                }
            )

            state["resolution"] = resolution

            # Add to messages
            status_msg = "Awaiting human approval" if state["requires_approval"] else "Ready to execute"
            state["messages"].append({
                "role": "system",
                "content": f"Resolution prepared: {status_msg}"
            })

            logger.info(f"[{state['trace_id']}] Resolution prepared: {resolution.status}")

            return state

        except Exception as e:
            logger.error(f"[{state['trace_id']}] Approval preparation failed: {e}")
            state["error"] = f"Approval preparation error: {e}"
            return state

    def _handle_error_node(self, state: OrchestratorState) -> OrchestratorState:
        """
        Node: Handle errors.

        Inputs from state:
        - error: Error message

        Outputs to state:
        - resolution: Error resolution
        """
        logger.error(f"[{state['trace_id']}] Handling error: {state.get('error')}")

        # Create error resolution
        resolution = Resolution(
            resolution_id=f"res_{state['trace_id']}_error",
            issue_id=state["issue"].issue_id,
            status="error",
            error_message=state.get("error"),
            created_at=datetime.now()
        )

        state["resolution"] = resolution

        return state

    # ========================================================================
    # CONDITIONAL ROUTING
    # ========================================================================

    def _route_decision(self, state: OrchestratorState) -> str:
        """
        Conditional edge: Decide whether to propose skill or go to novel path.

        LangGraph Concept: This is a CONDITIONAL EDGE function.
        It examines state and returns the name of the next node to visit.

        Returns:
        - "skill_proposal": If any skill match found (propose to human)
        - "novel_path": If no skill match found
        - "error": If error occurred
        """
        # Check for errors first
        if state.get("error"):
            logger.warning(f"[{state['trace_id']}] Routing to error handler")
            return "error"

        # Check if we have any skill match (regardless of confidence)
        best_match = state.get("best_match")

        if best_match:
            logger.info(f"[{state['trace_id']}] Routing to skill proposal (score: {best_match.score:.3f})")
            return "skill_proposal"
        else:
            logger.info(f"[{state['trace_id']}] Routing to novel path (no skill match)")
            return "novel_path"

    def _skill_approval_decision(self, state: OrchestratorState) -> str:
        """
        Conditional edge: Route based on skill proposal approval.

        Returns:
        - "skill_path": If human approved the skill
        - "novel_path": If human rejected the skill
        """
        if state.get("skill_proposal_approved"):
            logger.info(f"[{state['trace_id']}] Skill approved - routing to skill path")
            return "skill_path"
        else:
            reason = state.get("skill_rejection_reason", "No reason provided")
            logger.info(f"[{state['trace_id']}] Skill rejected ({reason}) - routing to TPAO")
            return "novel_path"

    # ========================================================================
    # PUBLIC API
    # ========================================================================

    def process_issue(self, issue: Issue) -> Resolution:
        """
        Process a customer issue through the orchestrator.

        This is the main entry point. It:
        1. Creates initial state
        2. Invokes the LangGraph workflow
        3. Returns the resolution

        Args:
            issue: Customer issue to process

        Returns:
            Resolution with proposed actions
        """
        # Create initial state
        trace_id = f"trace_{issue.issue_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        initial_state: OrchestratorState = {
            "issue": issue,
            "intent": None,
            "category": None,
            "priority": None,
            "retrieved_docs": [],
            "context_summary": None,
            "retrieval_confidence": None,
            "skill_matches": [],
            "best_match": None,
            "skill_proposal_approved": None,
            "skill_rejection_reason": None,
            "route": None,
            "proposed_actions": [],
            "requires_approval": False,
            "approval_reason": None,
            "resolution": None,
            "error": None,
            "retry_count": 0,
            "trace_id": trace_id,
            "start_time": datetime.now().isoformat(),
            "messages": []
        }

        logger.info(f"[{trace_id}] Starting orchestration for issue: {issue.issue_id}")

        try:
            # Invoke the LangGraph workflow
            # This runs through all nodes following the edges we defined
            final_state = self.graph.invoke(initial_state)

            # Extract resolution
            resolution = final_state.get("resolution")

            if not resolution:
                raise ValueError("No resolution generated")

            logger.info(f"[{trace_id}] Orchestration complete: {resolution.status}")

            return resolution

        except Exception as e:
            logger.error(f"[{trace_id}] Orchestration failed: {e}")

            # Create error resolution
            error_resolution = Resolution(
                resolution_id=f"res_{trace_id}_error",
                issue_id=issue.issue_id,
                status="error",
                error_message=str(e),
                created_at=datetime.now()
            )

            return error_resolution
