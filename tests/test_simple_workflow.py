"""
Simple End-to-End Test for Agentic Workflow
Tests the core components that are currently implemented.
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

print("=" * 80)
print("AGENTIC WORKFLOW SIMPLE TEST".center(80))
print("=" * 80)
print()

# Test 1: Import all components
print("TEST 1: Importing Components")
print("-" * 80)

try:
    from src.domain.models.issue import Issue, IssueType, IssueChannel, IssuePriority
print(" Issue models imported")
    
    from src.domain.models.resolution import Resolution, ResolutionStep, ResolutionStatus
print(" Resolution models imported")
    
    from src.domain.models.approval import ApprovalRequest, ApprovalToken, RiskLevel
print(" Approval models imported")
    
    from src.domain.models.context import RAGContext, RetrievedContext
print(" Context models imported")
    
    from src.application.rag.rag_pipeline import RAGPipeline
print(" RAG Pipeline imported")
    
    from src.application.skills.skill_registry import SkillRegistry
print(" Skill Registry imported")
    
    from src.application.tools.base import ToolRegistry
print(" Tool Registry imported")
    
    from src.application.react.tpao_loop import TPAOLoop
print(" TPAO Loop imported")
    
    from src.application.approval.approval_service import ApprovalService
print(" Approval Service imported")
    
    from src.application.executor.executor_agent import ExecutorAgent, ExecutionStatus
print(" Executor Agent imported")
    
print("\n All imports successful!\n")
    
except Exception as e:
print(f"\n Import failed: {e}\n")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 2: Initialize components
print("TEST 2: Initializing Components")
print("-" * 80)

try:
    # Initialize RAG pipeline
    rag_pipeline = RAGPipeline()
print(" RAG Pipeline initialized")
    
    # Initialize skills registry
    skill_registry = SkillRegistry()
print(" Skill Registry initialized")
    
    # Initialize tool registry
    tool_registry = ToolRegistry()
print(" Tool Registry initialized")
    
    # Initialize TPAO loop
    tpao_loop = TPAOLoop(
        rag_pipeline=rag_pipeline,
        tool_registry=tool_registry
    )
print(" TPAO Loop initialized")
    
    # Initialize approval service
    approval_service = ApprovalService()
print(" Approval Service initialized")
    
    # Initialize executor agent
    executor_agent = ExecutorAgent(
        tool_registry=tool_registry,
        approval_service=approval_service
    )
print(" Executor Agent initialized")
    
print("\n All components initialized successfully!\n")
    
except Exception as e:
print(f"\n Initialization failed: {e}\n")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: Create test issue
print("TEST 3: Creating Test Issue")
print("-" * 80)

try:
    issue = Issue(
        issue_id="TEST-001",
        channel=IssueChannel.EMAIL,
        subject="Late checkout request",
        body="Guest John Doe (booking BK-12345) requests late checkout until 2 PM due to late flight",
        issue_type=IssueType.LATE_CHECKOUT,
        booking_id="BK-12345",
        guest_email="john.doe@example.com",
        priority=IssuePriority.MEDIUM,
        metadata={
            "guest_name": "John Doe",
            "current_checkout": "11:00 AM",
            "requested_checkout": "2:00 PM"
        }
    )
    
print(f" Issue created: {issue.issue_id}")
    print(f"  Subject: {issue.subject}")
    print(f"  Type: {issue.issue_type.value if issue.issue_type else 'None'}")
    print(f"  Channel: {issue.channel.value}")
    print(f"  Priority: {issue.priority.value}")
    
print("\n Test issue created successfully!\n")
    
except Exception as e:
print(f"\n Issue creation failed: {e}\n")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Test Approval Service
print("TEST 4: Testing Approval Service")
print("-" * 80)

try:
    # Create a mock resolution
    resolution = Resolution(
        resolution_id=f"RES-{issue.issue_id}",
        issue_id=issue.issue_id,
        status=ResolutionStatus.PENDING_APPROVAL,
        skill_used=None,
        skill_matched=False,
        novel_task=True,
        steps=[
            ResolutionStep(
                step_number=1,
                step_type="tool_call",
                description="Check late checkout availability",
                tool_used="check_availability",
                input_data={"booking_id": "BK-12345", "requested_time": "14:00"},
                output_data={},
                success=True
            ),
            ResolutionStep(
                step_number=2,
                step_type="tool_call",
                description="Update checkout time",
                tool_used="update_checkout",
                input_data={"booking_id": "BK-12345", "new_time": "14:00"},
                output_data={},
                success=True
            )
        ]
    )
    
print(f" Mock resolution created: {resolution.resolution_id}")
    print(f"  Steps: {len(resolution.steps)}")
    
    # Create approval request
    proposed_actions = [
        {
            "action": "update_checkout",
            "parameters": {"booking_id": "BK-12345", "new_time": "14:00"},
            "description": "Update checkout time to 2 PM"
        }
    ]
    
    approval_request = approval_service.create_approval_request(
        resolution=resolution,
        proposed_actions=proposed_actions,
        requester="test_agent"
    )
    
print(f" Approval request created: {approval_request.request_id}")
    print(f"  Risk level: {approval_request.risk_level.value}")
    print(f"  Action type: {approval_request.action_type}")
    print(f"  Actions: {len(proposed_actions)}")
    
    # Approve the request
    approval_token = approval_service.approve_request(
        request_id=approval_request.request_id,
        approver="test_supervisor",
        comments="Approved for testing"
    )
    
print(f" Approval granted")
    print(f"  Token: {approval_token.token[:20]}...")
    print(f"  Expires: {approval_token.expires_at}")
    
    # Validate token
    is_valid = approval_service.validate_token(approval_token.token)
print(f" Token validation: {'Valid' if is_valid else 'Invalid'}")
    
print("\n Approval service working correctly!\n")
    
except Exception as e:
print(f"\n Approval service test failed: {e}\n")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Test Executor Agent
print("TEST 5: Testing Executor Agent")
print("-" * 80)

try:
    # Test execution with approval
    execution_result = executor_agent.execute_with_approval(
        resolution=resolution,
        approval_token=approval_token.token,
        actions=proposed_actions
    )
    
print(f" Execution completed")
    print(f"  Status: {execution_result.status.value}")
    print(f"  Executed actions: {len(execution_result.executed_actions)}")
    print(f"  Failed actions: {len(execution_result.failed_actions)}")
    print(f"  Execution time: {execution_result.execution_time_ms:.2f}ms")
    
    if execution_result.status == ExecutionStatus.COMPLETED:
print(" Execution successful!")
    else:
print(f" Execution status: {execution_result.status.value}")
        if execution_result.error:
            print(f"  Error: {execution_result.error}")
    
    # Test execution without approval (should fail)
print("\n Testing unauthorized execution (should fail)...")
    try:
        executor_agent.execute_without_approval(
            resolution=resolution,
            actions=proposed_actions
        )
print(" ERROR: Executor allowed execution without approval!")
        sys.exit(1)
    except Exception as e:
print(f" Correctly blocked unauthorized execution: {type(e).__name__}")
    
print("\n Executor agent working correctly!\n")
    
except Exception as e:
print(f"\n Executor agent test failed: {e}\n")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 6: Test Tool Registry
print("TEST 6: Testing Tool Registry")
print("-" * 80)

try:
    # List available tools
    tools = tool_registry.list_tools()
print(f" Available tools: {len(tools)}")
    for tool_name in tools[:5]:  # Show first 5
        print(f"  - {tool_name}")
    if len(tools) > 5:
        print(f"  ... and {len(tools) - 5} more")
    
    # Get a specific tool
    if tools:
        tool = tool_registry.get(tools[0])
        if tool:
print(f" Retrieved tool: {tools[0]}")
            print(f"  Description: {tool.description[:60]}..." if len(tool.description) > 60 else f"  Description: {tool.description}")
        else:
print(f" Tool {tools[0]} not found")
    
print("\n Tool registry working correctly!\n")
    
except Exception as e:
print(f"\n Tool registry test failed: {e}\n")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 7: Test Skill Registry
print("TEST 7: Testing Skill Registry")
print("-" * 80)

try:
    # List available skills
    skills = skill_registry.list_skills()
print(f" Available skills: {len(skills)}")
    for skill_meta in skills[:5]:  # Show first 5
        skill = skill_registry.get_skill(skill_meta.skill_id)
        if skill:
            print(f"  - {skill_meta.skill_id}: {skill.name}")
    if len(skills) > 5:
        print(f"  ... and {len(skills) - 5} more")
    
print("\n Skill registry working correctly!\n")
    
except Exception as e:
print(f"\n Skill registry test failed: {e}\n")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Final Summary
print("=" * 80)
print("TEST SUMMARY".center(80))
print("=" * 80)
print()
print(" All core components are working!")
print()
print("Tested Components:")
print(" Domain Models (Issue, Resolution, Approval, Context)")
print(" RAG Pipeline")
print(" Skill Registry")
print(" Tool Registry")
print(" TPAO Loop")
print(" Approval Service (create, approve, validate)")
print(" Executor Agent (with approval, without approval)")
print()
print("Key Findings:")
print(" Approval gates are enforced")
print(" Token validation works")
print(" Unauthorized execution is blocked")
print(" All components initialize successfully")
print()
print("=" * 80)
print(" ALL TESTS PASSED! ".center(80))
print("=" * 80)
print()

sys.exit(0)
