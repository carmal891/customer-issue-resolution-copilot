"""
Skill Compiler

Converts successful ReAct/TPAO traces into reusable YAML skill definitions.
This is the "learning" mechanism that allows the system to improve over time.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import yaml
import logging
import re

from src.domain.models.skill import Skill, SkillStatus
from src.domain.models.issue import Issue
from src.domain.models.resolution import Resolution
from src.infrastructure.llm.llm_service import LLMService

logger = logging.getLogger(__name__)


@dataclass
class ReActTrace:
    """
    Trace of a ReAct/TPAO execution.

    Captures the full execution history for skill compilation.
    """
    issue: Issue
    resolution: Resolution
    think_steps: List[Dict[str, Any]]
    plan_steps: List[Dict[str, Any]]
    act_steps: List[Dict[str, Any]]
    observe_steps: List[Dict[str, Any]]
    tools_used: List[str]
    approval_required: List[str]
    execution_time_ms: float
    success: bool
    metadata: Dict[str, Any]


class SkillCompilationError(Exception):
    """Raised when skill compilation fails"""
    pass


class SkillCompiler:
    """
    Compiles ReAct traces into reusable skills.

    Compilation Process:
    1. Extract trigger patterns from issue description
    2. Identify reusable steps from trace
    3. Determine approval gates
    4. Extract guardrails and constraints
    5. Generate YAML skill definition
    6. Validate skill structure

    Responsibilities:
    - Analyze successful ReAct traces
    - Extract generalizable patterns
    - Generate skill YAML files
    - Validate skill structure
    - Save draft skills for human review
    """

    def __init__(
        self,
        skills_dir: str = "data/skills",
        draft_dir: str = "data/skills/drafts",
        llm_service: Optional[LLMService] = None
    ):
        """
        Initialize skill compiler.

        Args:
            skills_dir: Directory for approved skills
            draft_dir: Directory for draft skills awaiting review
            llm_service: LLM service for generating rich descriptions (optional)
        """
        self.skills_dir = Path(skills_dir)
        self.draft_dir = Path(draft_dir)
        self.llm_service = llm_service

        # Create directories if they don't exist
        self.draft_dir.mkdir(parents=True, exist_ok=True)

    def compile_skill(
        self,
        trace: ReActTrace,
        skill_name: Optional[str] = None,
        skill_id: Optional[str] = None
    ) -> Skill:
        """
        Compile a ReAct trace into a skill.

        Args:
            trace: ReAct execution trace
            skill_name: Optional custom skill name
            skill_id: Optional custom skill ID

        Returns:
            Compiled Skill instance

        Raises:
            SkillCompilationError: If compilation fails
        """
        try:
            # Generate skill ID and name
            if not skill_id:
                skill_id = self._generate_skill_id(trace)
            if not skill_name:
                skill_name = self._generate_skill_name(trace)

            # Extract components
            triggers = self._extract_triggers(trace)
            steps = self._extract_steps(trace)
            approval_gates = self._extract_approval_gates(trace)
            guardrails = self._extract_guardrails(trace)
            metadata = self._build_metadata(trace)

            # Create Skill instance
            skill = Skill(
                skill_id=skill_id,
                name=skill_name,
                version="1.0",
                status=SkillStatus.DRAFT,
                description=self._generate_description(trace),
                triggers=triggers,
                steps=steps,
                guardrails=guardrails,
                metadata=metadata,
                created_by="skill_compiler"
            )

            logger.info(f"Compiled skill: {skill_id} from trace")
            return skill

        except Exception as e:
            logger.error(f"Failed to compile skill: {e}")
            raise SkillCompilationError(f"Failed to compile skill: {e}")

    def save_draft_skill(self, skill: Skill) -> str:
        """
        Save skill as draft YAML file.

        Args:
            skill: Skill to save

        Returns:
            Path to saved file
        """
        try:
            # Convert skill to YAML-compatible dict
            skill_dict = self._skill_to_yaml_dict(skill)

            # Generate filename
            filename = f"{skill.skill_id}.yaml"
            filepath = self.draft_dir / filename

            # Write to file
            with open(filepath, 'w') as f:
                yaml.dump(skill_dict, f, default_flow_style=False, sort_keys=False)

            logger.info(f"Saved draft skill to: {filepath}")
            return str(filepath)

        except Exception as e:
            logger.error(f"Failed to save draft skill: {e}")
            raise SkillCompilationError(f"Failed to save draft skill: {e}")

    def _generate_skill_id(self, trace: ReActTrace) -> str:
        """
        Generate meaningful skill ID from trace using LLM.
        
        Creates a snake_case ID that describes what the skill does.
        Falls back to category-based ID if LLM is not available.
        
        Examples:
        - late_checkout_request
        - room_upgrade_processing
        - billing_dispute_resolution
        """
        # Fallback if no LLM service
        if not self.llm_service:
            category = trace.issue.metadata.get('category', 'general')
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            category_clean = re.sub(r'[^a-z0-9_]', '_', category.lower())
            return f"{category_clean}_skill_{timestamp}"
        
        try:
            # Build context for ID generation
            issue_summary = trace.issue.subject or trace.issue.body[:200]
            tools_used = ", ".join(trace.tools_used[:2]) if trace.tools_used else "general actions"
            
            # Build LLM prompt for ID generation
            id_prompt = f"""You are generating a concise, descriptive skill ID (filename) for a customer service skill.

**Customer Issue:** {issue_summary}

**Tools/Actions Used:** {tools_used}

**Task:** Generate a short skill ID in snake_case format (2-4 words) that clearly indicates what this skill does.

**Examples of good skill IDs:**
- late_checkout_request
- room_upgrade_processing
- billing_dispute_resolution
- amenity_request_fulfillment
- cancellation_refund_processing
- guest_complaint_handling

**Requirements:**
- Use snake_case (lowercase with underscores)
- 2-4 words maximum
- Be specific and descriptive
- No timestamps or random characters
- Focus on the ACTION or OUTCOME

Generate ONLY the skill ID, nothing else."""

            # Get LLM-generated ID
            llm_response = self.llm_service.generate_simple(id_prompt)
            skill_id = llm_response.content.strip().lower()
            
            # Clean up the ID (remove quotes, extra whitespace, ensure snake_case)
            skill_id = skill_id.strip('"\'').strip()
            skill_id = re.sub(r'[^a-z0-9_]', '_', skill_id)
            skill_id = re.sub(r'_+', '_', skill_id)  # Remove multiple underscores
            skill_id = skill_id.strip('_')  # Remove leading/trailing underscores
            
            # Validate ID (should be 2-4 words, roughly 10-40 chars)
            word_count = len(skill_id.split('_'))
            if word_count < 2 or word_count > 5 or len(skill_id) > 50 or len(skill_id) < 5:
                logger.warning(f"LLM generated invalid skill ID ('{skill_id}'), using fallback")
                category = trace.issue.metadata.get('category', 'general')
                category_clean = re.sub(r'[^a-z0-9_]', '_', category.lower())
                return f"{category_clean}_handler"
            
            logger.info(f"Generated LLM-powered skill ID: '{skill_id}'")
            return skill_id
            
        except Exception as e:
            logger.error(f"Failed to generate LLM skill ID: {e}, using fallback")
            category = trace.issue.metadata.get('category', 'general')
            category_clean = re.sub(r'[^a-z0-9_]', '_', category.lower())
            return f"{category_clean}_handler"

    def _generate_skill_name(self, trace: ReActTrace) -> str:
        """
        Generate meaningful, human-readable skill name using LLM.
        
        Creates a concise name that captures the essence of what the skill does.
        Falls back to simple name if LLM is not available.
        """
        # Fallback if no LLM service
        if not self.llm_service:
            category = trace.issue.metadata.get('category', 'General')
            return f"{category} Handler (Auto-generated)"
        
        try:
            # Build context for name generation
            issue_summary = trace.issue.subject or trace.issue.body[:200]
            tools_used = ", ".join(trace.tools_used[:3]) if trace.tools_used else "general actions"
            
            # Build LLM prompt for name generation
            name_prompt = f"""You are generating a concise, professional name for a customer service skill.

**Customer Issue:** {issue_summary}

**Tools/Actions Used:** {tools_used}

**Task:** Generate a short, descriptive skill name (3-6 words) that clearly indicates what this skill does.

**Examples of good skill names:**
- "Late Checkout Request Handler"
- "Room Upgrade Processing"
- "Billing Dispute Resolution"
- "Amenity Request Fulfillment"
- "Cancellation and Refund Processing"

**Requirements:**
- Use title case
- Be specific and actionable
- 3-6 words maximum
- No generic terms like "Handler" unless necessary
- Focus on the ACTION or OUTCOME

Generate ONLY the skill name, nothing else."""

            # Get LLM-generated name
            llm_response = self.llm_service.generate_simple(name_prompt)
            skill_name = llm_response.content.strip()
            
            # Clean up the name (remove quotes, extra whitespace)
            skill_name = skill_name.strip('"\'').strip()
            
            # Validate name length (should be 3-6 words, roughly 15-50 chars)
            word_count = len(skill_name.split())
            if word_count < 2 or word_count > 8 or len(skill_name) > 80:
                logger.warning(f"LLM generated invalid skill name ('{skill_name}'), using fallback")
                category = trace.issue.metadata.get('category', 'General')
                return f"{category} Handler"
            
            logger.info(f"Generated LLM-powered skill name: '{skill_name}'")
            return skill_name
            
        except Exception as e:
            logger.error(f"Failed to generate LLM skill name: {e}, using fallback")
            category = trace.issue.metadata.get('category', 'General')
            return f"{category} Handler"

    def _generate_description(self, trace: ReActTrace) -> str:
        """
        Generate rich, meaningful skill description using LLM.
        
        Creates a description that includes:
        - Problem summary
        - Solution approach
        - Key actions/tools used
        - Policy context
        - When to use this skill
        
        Falls back to simple description if LLM is not available.
        """
        # Fallback if no LLM service
        if not self.llm_service:
            issue_summary = trace.issue.subject or trace.issue.body[:100]
            return f"Auto-generated skill for handling: {issue_summary}"
        
        try:
            # Build context from trace
            issue_context = f"""
Issue Channel: {trace.issue.channel}
Issue Subject: {trace.issue.subject or 'N/A'}
Issue Body: {trace.issue.body[:300]}
"""
            
            # Extract key information from trace steps
            tools_used = ", ".join(trace.tools_used) if trace.tools_used else "None"
            num_steps = len(trace.act_steps)
            requires_approval = "Yes" if trace.approval_required else "No"
            
            # Get context from Think phase (RAG-retrieved policies)
            policy_context = ""
            if trace.think_steps:
                for think_step in trace.think_steps:
                    if 'output_data' in think_step and 'citations' in think_step['output_data']:
                        citations = think_step['output_data']['citations']
                        if citations:
                            policy_context = f"Referenced policies: {', '.join([c.get('source_name', 'Unknown') for c in citations[:3]])}"
                            break
            
            # Build LLM prompt for description generation
            description_prompt = f"""You are generating a concise, meaningful description for a customer service skill that was automatically learned from a successful resolution.

**Customer Issue:**
{issue_context}

**Resolution Details:**
- Number of steps: {num_steps}
- Tools used: {tools_used}
- Requires approval: {requires_approval}
- {policy_context}

**Task:** Generate a 2-4 sentence skill description that includes:
1. What customer problem this skill solves
2. How it solves it (key actions)
3. What tools/systems it uses
4. Any important constraints or approval requirements

**Example good description:**
"Handles guest requests for extended checkout times beyond the standard 11 AM deadline. Checks room availability and housekeeping schedule, then processes late checkout approval with appropriate fees based on hotel policy. Uses lookup_booking and update_checkout_time tools. Requires manager approval for extensions beyond 2 PM."

Generate a similar description for this skill. Be specific and actionable. Focus on WHAT it does and HOW it works."""

            # Get LLM-generated description
            llm_response = self.llm_service.generate_simple(description_prompt)
            description = llm_response.content.strip()
            
            # Validate description length (should be 2-4 sentences, roughly 100-400 chars)
            if len(description) < 50:
                logger.warning(f"LLM generated too short description ({len(description)} chars), using fallback")
                issue_summary = trace.issue.subject or trace.issue.body[:100]
                return f"Auto-generated skill for handling: {issue_summary}"
            
            if len(description) > 600:
                logger.warning(f"LLM generated too long description ({len(description)} chars), truncating")
                description = description[:597] + "..."
            
            logger.info(f"Generated LLM-powered skill description ({len(description)} chars)")
            return description
            
        except Exception as e:
            logger.error(f"Failed to generate LLM description: {e}, using fallback")
            issue_summary = trace.issue.subject or trace.issue.body[:100]
            return f"Auto-generated skill for handling: {issue_summary}"

    def _extract_triggers(self, trace: ReActTrace) -> List[str]:
        """
        Extract trigger patterns from issue.

        Uses issue title, description, and keywords to identify patterns.
        """
        triggers = []

        # Add issue title as primary trigger
        if trace.issue.title:
            triggers.append(trace.issue.title.lower())

        # Extract key phrases from description
        if trace.issue.description:
            # Simple keyword extraction (can be enhanced with NLP)
            desc_lower = trace.issue.description.lower()

            # Common patterns
            patterns = [
                r'(request|need|want|require)\s+(\w+(?:\s+\w+){0,3})',
                r'(issue|problem|error)\s+with\s+(\w+(?:\s+\w+){0,2})',
                r'(cannot|can\'t|unable to)\s+(\w+(?:\s+\w+){0,2})'
            ]

            for pattern in patterns:
                matches = re.findall(pattern, desc_lower)
                for match in matches:
                    phrase = ' '.join(match).strip()
                    if phrase and len(phrase) > 5:
                        triggers.append(phrase)

        # Add category as trigger
        category = trace.issue.metadata.get('category')
        if category:
            triggers.append(category.lower())

        # Deduplicate and limit
        triggers = list(set(triggers))[:10]

        return triggers if triggers else ["general issue"]

    def _extract_steps(self, trace: ReActTrace) -> List[Dict[str, Any]]:
        """
        Extract reusable steps from trace.

        Converts ReAct actions into skill steps.
        """
        steps = []

        # Combine plan and act steps
        for i, act_step in enumerate(trace.act_steps):
            step = {
                'step_id': f"step_{i+1}",
                'step_type': act_step.get('type', 'tool_call'),
                'description': act_step.get('description', f"Step {i+1}"),
                'tool_name': act_step.get('tool_name'),
                'parameters': act_step.get('parameters', {}),
                'requires_approval': act_step.get('tool_name') in trace.approval_required,
                'expected_output': act_step.get('expected_output')
            }
            steps.append(step)

        return steps

    def _extract_approval_gates(self, trace: ReActTrace) -> List[Dict[str, Any]]:
        """Extract approval gate requirements"""
        gates = []

        for tool_name in trace.approval_required:
            gate = {
                'gate_id': f"approval_{tool_name}",
                'trigger': f"before_{tool_name}",
                'reason': f"Requires approval for {tool_name}",
                'risk_level': "medium"
            }
            gates.append(gate)

        return gates

    def _extract_guardrails(self, trace: ReActTrace) -> Dict[str, Any]:
        """Extract guardrails and constraints"""
        guardrails = {
            'max_retries': 3,
            'timeout_seconds': 300,
            'requires_approval': len(trace.approval_required) > 0
        }

        # Add domain-specific guardrails from metadata
        if 'guardrails' in trace.metadata:
            guardrails.update(trace.metadata['guardrails'])

        return guardrails

    def _build_metadata(self, trace: ReActTrace) -> Dict[str, Any]:
        """Build skill metadata"""
        return {
            'domain': trace.issue.metadata.get('domain', 'general'),
            'category': trace.issue.metadata.get('category', 'general'),
            'created_from': 'react_trace',
            'source_issue_id': trace.issue.issue_id,
            'compilation_date': datetime.now().isoformat(),
            'avg_execution_time_ms': trace.execution_time_ms,
            'tools_used': trace.tools_used,
            'success_rate': 1.0 if trace.success else 0.0
        }

    def _skill_to_yaml_dict(self, skill: Skill) -> Dict[str, Any]:
        """
        Convert Skill to YAML-compatible dictionary.

        Matches the YAML schema used in seed skills.
        """
        return {
            'skill_id': skill.skill_id,
            'version': skill.version,
            'name': skill.name,
            'description': skill.description,
            'metadata': {
                'domain': skill.metadata.get('domain', 'general'),
                'category': skill.metadata.get('category', 'general'),
                'created_from': skill.metadata.get('created_from', 'unknown'),
                'created_at': skill.created_at.isoformat(),
                'created_by': skill.created_by,
                'tags': []
            },
            'triggers': {
                'semantic_patterns': skill.triggers,
                'keywords': skill.triggers[:5],  # Use first 5 as keywords
                'intent_categories': [skill.metadata.get('category', 'general')]
            },
            'steps': skill.steps,
            'approval_gates': [
                {
                    'gate_id': 'approval_auto_generated',
                    'trigger': 'before_execution',
                    'reason': 'Auto-generated skill requires review',
                    'risk_level': 'medium'
                }
            ] if skill.requires_human_approval() else [],
            'guardrails': skill.guardrails,
            'error_handling': [
                {
                    'error_type': 'tool_execution_error',
                    'action': 'retry',
                    'max_retries': 3
                },
                {
                    'error_type': 'timeout',
                    'action': 'escalate',
                    'escalation_target': 'human_operator'
                }
            ],
            'success_criteria': [
                'All steps completed successfully',
                'No errors encountered',
                'Customer issue resolved'
            ],
            'rollback_steps': [],
            'source_trace': {
                'trace_id': skill.metadata.get('source_issue_id'),
                'compilation_date': skill.metadata.get('compilation_date')
            },
            'performance_metrics': {
                'avg_execution_time_ms': skill.metadata.get('avg_execution_time_ms', 0),
                'success_rate': skill.metadata.get('success_rate', 0.0),
                'usage_count': 0
            }
        }

    def validate_skill(self, skill: Skill) -> bool:
        """
        Validate skill structure.

        Args:
            skill: Skill to validate

        Returns:
            True if valid

        Raises:
            SkillCompilationError: If validation fails
        """
        errors = []

        # Check required fields
        if not skill.skill_id:
            errors.append("Missing skill_id")
        if not skill.name:
            errors.append("Missing name")
        if not skill.description:
            errors.append("Missing description")
        if not skill.triggers:
            errors.append("Missing triggers")

        # Check steps
        if not skill.steps:
            errors.append("No steps defined")

        if errors:
            raise SkillCompilationError(f"Skill validation failed: {', '.join(errors)}")

        return True
