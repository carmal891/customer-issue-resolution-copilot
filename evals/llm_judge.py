"""
LLM Judge - Base class for LLM-as-a-Judge evaluation

Uses GPT-4 or Claude to evaluate system outputs with structured prompts.
Provides reusable judge methods for faithfulness, relevancy, plan quality, etc.
"""

from typing import Dict, Any, Optional, List, Literal
from dataclasses import dataclass
from enum import Enum
import json
import logging
from openai import OpenAI
import os

logger = logging.getLogger(__name__)


class JudgmentType(str, Enum):
    """Types of judgments the LLM can make"""
    FAITHFULNESS = "faithfulness"
    ANSWER_RELEVANCY = "answer_relevancy"
    PLAN_QUALITY = "plan_quality"
    TOOL_CORRECTNESS = "tool_correctness"
    GUARDRAIL_EFFECTIVENESS = "guardrail_effectiveness"


@dataclass
class JudgmentResult:
    """Result of an LLM judgment"""
    judgment_type: str
    score: float  # 0.0 to 1.0
    reasoning: str
    verdict: str  # "pass", "fail", "partial"
    metadata: Dict[str, Any]


class LLMJudge:
    """
    LLM-as-a-Judge for evaluating system outputs.
    
    Uses structured prompts with temperature=0 for reproducibility.
    Supports multiple judgment types with specific evaluation criteria.
    """
    
    # Prompt templates for different judgment types
    FAITHFULNESS_PROMPT = """You are an expert evaluator assessing whether an AI assistant's answer is faithful to the provided context.

**Task**: Determine if the answer contains any information that is NOT supported by the context.

**Context**:
{context}

**Question**:
{question}

**Answer**:
{answer}

**Evaluation Criteria**:
1. Every claim in the answer must be directly supported by the context
2. The answer should not add information beyond what's in the context
3. Paraphrasing is acceptable if the meaning is preserved
4. If the answer says "I don't know" or similar, check if the context actually lacks the information

**Output Format** (JSON):
{{
    "score": <float between 0.0 and 1.0>,
    "verdict": "<pass|fail|partial>",
    "reasoning": "<detailed explanation>",
    "unsupported_claims": [<list of claims not supported by context>]
}}

Provide your evaluation:"""

    ANSWER_RELEVANCY_PROMPT = """You are an expert evaluator assessing whether an AI assistant's answer is relevant to the user's question.

**Question**:
{question}

**Answer**:
{answer}

**Evaluation Criteria**:
1. Does the answer directly address the question asked?
2. Is the answer focused and on-topic?
3. Does it avoid unnecessary tangents or irrelevant information?
4. If the question cannot be answered, does it explain why clearly?

**Output Format** (JSON):
{{
    "score": <float between 0.0 and 1.0>,
    "verdict": "<pass|fail|partial>",
    "reasoning": "<detailed explanation>",
    "relevance_issues": [<list of relevance problems if any>]
}}

Provide your evaluation:"""

    PLAN_QUALITY_PROMPT = """You are an expert evaluator assessing the quality of a resolution plan for a customer issue.

**Customer Issue**:
{issue}

**Resolution Plan**:
{plan}

**Retrieved Context** (policies, procedures):
{context}

**Evaluation Criteria** (1-5 scale):
1. **Completeness**: Does the plan address all aspects of the issue?
2. **Correctness**: Are the steps aligned with company policies?
3. **Clarity**: Are the steps clear and actionable?
4. **Efficiency**: Is the plan reasonably efficient (not overly complex)?
5. **Safety**: Does it include appropriate approval gates for risky actions?

**Output Format** (JSON):
{{
    "overall_score": <float between 1.0 and 5.0>,
    "completeness": <1-5>,
    "correctness": <1-5>,
    "clarity": <1-5>,
    "efficiency": <1-5>,
    "safety": <1-5>,
    "verdict": "<excellent|good|acceptable|poor>",
    "reasoning": "<detailed explanation>",
    "improvement_suggestions": [<list of suggestions>]
}}

Provide your evaluation:"""

    TOOL_CORRECTNESS_PROMPT = """You are an expert evaluator assessing whether the correct tools were selected for a task.

**Task Description**:
{task}

**Available Tools**:
{available_tools}

**Selected Tools**:
{selected_tools}

**Tool Execution Results**:
{tool_results}

**Evaluation Criteria**:
1. Were the right tools selected for the task?
2. Were tools used in the correct order?
3. Were tool parameters appropriate?
4. Were any necessary tools missed?
5. Were any unnecessary tools used?

**Output Format** (JSON):
{{
    "score": <float between 0.0 and 1.0>,
    "verdict": "<pass|fail|partial>",
    "reasoning": "<detailed explanation>",
    "correct_tools": [<list of correctly used tools>],
    "incorrect_tools": [<list of incorrectly used tools>],
    "missing_tools": [<list of tools that should have been used>]
}}

Provide your evaluation:"""

    def __init__(
        self,
        model: str = "gpt-5.4-mini",
        temperature: float = 0.0,
        api_key: Optional[str] = None
    ):
        """
        Initialize LLM Judge.
        
        Args:
            model: OpenAI model to use
                   Recommended options:
                   - "gpt-5.4-mini" (default): Latest model with improved reasoning
                   - "gpt-5.4-nano": Faster/cheaper for simpler judgments
                   - "gpt-4o-mini": Older model (may not be available)
            temperature: Temperature for generation (default: 0.0 for reproducibility)
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
        """
        self.model = model
        self.temperature = temperature
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        
    def judge_faithfulness(
        self,
        question: str,
        answer: str,
        context: str
    ) -> JudgmentResult:
        """
        Judge if answer is faithful to context (no hallucinations).
        
        Args:
            question: The question asked
            answer: The answer provided
            context: The context used to generate the answer
            
        Returns:
            JudgmentResult with faithfulness score (0.0-1.0)
        """
        prompt = self.FAITHFULNESS_PROMPT.format(
            question=question,
            answer=answer,
            context=context
        )
        
        return self._call_judge(
            prompt=prompt,
            judgment_type=JudgmentType.FAITHFULNESS
        )
    
    def judge_answer_relevancy(
        self,
        question: str,
        answer: str
    ) -> JudgmentResult:
        """
        Judge if answer is relevant to the question.
        
        Args:
            question: The question asked
            answer: The answer provided
            
        Returns:
            JudgmentResult with relevancy score (0.0-1.0)
        """
        prompt = self.ANSWER_RELEVANCY_PROMPT.format(
            question=question,
            answer=answer
        )
        
        return self._call_judge(
            prompt=prompt,
            judgment_type=JudgmentType.ANSWER_RELEVANCY
        )
    
    def judge_plan_quality(
        self,
        issue: str,
        plan: str,
        context: str
    ) -> JudgmentResult:
        """
        Judge the quality of a resolution plan (1-5 scale).
        
        Args:
            issue: The customer issue
            plan: The proposed resolution plan
            context: Retrieved policies/procedures used
            
        Returns:
            JudgmentResult with plan quality score (1.0-5.0)
        """
        prompt = self.PLAN_QUALITY_PROMPT.format(
            issue=issue,
            plan=plan,
            context=context
        )
        
        return self._call_judge(
            prompt=prompt,
            judgment_type=JudgmentType.PLAN_QUALITY
        )
    
    def judge_tool_correctness(
        self,
        task: str,
        available_tools: List[Dict[str, Any]],
        selected_tools: List[str],
        tool_results: Optional[Dict[str, Any]] = None
    ) -> JudgmentResult:
        """
        Judge if correct tools were selected and used.
        
        Args:
            task: The task description
            available_tools: List of available tools with descriptions
            selected_tools: List of tools that were selected
            tool_results: Results from tool execution (optional)
            
        Returns:
            JudgmentResult with tool correctness score (0.0-1.0)
        """
        prompt = self.TOOL_CORRECTNESS_PROMPT.format(
            task=task,
            available_tools=json.dumps(available_tools, indent=2),
            selected_tools=json.dumps(selected_tools, indent=2),
            tool_results=json.dumps(tool_results or {}, indent=2)
        )
        
        return self._call_judge(
            prompt=prompt,
            judgment_type=JudgmentType.TOOL_CORRECTNESS
        )
    
    def _call_judge(
        self,
        prompt: str,
        judgment_type: JudgmentType
    ) -> JudgmentResult:
        """
        Call LLM to make a judgment.
        
        Args:
            prompt: The evaluation prompt
            judgment_type: Type of judgment being made
            
        Returns:
            JudgmentResult with score and reasoning
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert evaluator. Provide objective, detailed evaluations in valid JSON format."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=self.temperature,
                response_format={"type": "json_object"}
            )
            
            # Parse JSON response
            result_text = response.choices[0].message.content
            if not result_text:
                raise ValueError("Empty response from LLM")
            result_data = json.loads(result_text)
            
            # Extract score (normalize to 0.0-1.0 if needed)
            score = result_data.get("score", 0.0)
            if judgment_type == JudgmentType.PLAN_QUALITY:
                # Plan quality is 1-5, normalize to 0-1
                score = (score - 1.0) / 4.0
            
            return JudgmentResult(
                judgment_type=judgment_type.value,
                score=float(score),
                reasoning=result_data.get("reasoning", ""),
                verdict=result_data.get("verdict", "unknown"),
                metadata=result_data
            )
            
        except Exception as e:
            logger.error(f"LLM judge call failed: {e}")
            return JudgmentResult(
                judgment_type=judgment_type.value,
                score=0.0,
                reasoning=f"Evaluation failed: {str(e)}",
                verdict="error",
                metadata={"error": str(e)}
            )
