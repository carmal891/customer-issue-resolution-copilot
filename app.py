"""
Customer Issue Resolution Copilot - Streamlit UI

A self-learning agent that turns scattered company knowledge into executable,
human-approved skills where every novel task it handles becomes a reusable skill.
"""

import streamlit as st
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List
import json
import os
import logging

# Load environment variables FIRST
from dotenv import load_dotenv
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Verify API key is loaded
if not os.getenv("OPENAI_API_KEY"):
    st.error("❌ OPENAI_API_KEY not found in environment variables. Please check your .env file.")
    st.stop()

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.domain.models.issue import Issue, IssueType, IssueChannel, IssuePriority
from src.domain.models.resolution import Resolution, ResolutionStep, ResolutionStatus
from src.domain.models.approval import ApprovalRequest, ApprovalStatus, RiskLevel
from src.application.rag.rag_pipeline import RAGPipeline
from src.application.skills.skill_registry import SkillRegistry
from src.application.skills.skill_matcher import SkillMatcher
from src.application.tools.base import ToolRegistry
from src.application.react.tpao_loop import TPAOLoop
from src.application.approval.approval_service import ApprovalService
from src.application.executor.executor_agent import ExecutorAgent
from src.infrastructure.llm.llm_service import LLMService
from src.application.guardrails import get_pii_detector, get_injection_detector

# Page config
st.set_page_config(
    page_title="Customer Issue Resolution Copilot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .error-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'initialized' not in st.session_state:
    st.session_state.initialized = False
    st.session_state.rag_pipeline = None
    st.session_state.skill_registry = None
    st.session_state.tool_registry = None
    st.session_state.tpao_loop = None
    st.session_state.approval_service = None
    st.session_state.executor_agent = None
    st.session_state.issues = []
    st.session_state.resolutions = []
    st.session_state.approvals = []

@st.cache_resource(show_spinner=False)
def initialize_system_components():
    """Initialize all system components (cached for performance)."""
    logger.info("Starting system initialization...")
    
    # Initialize RAG Pipeline
    logger.info("Initializing RAG Pipeline...")
    rag_pipeline = RAGPipeline()
    
    # Index policy documents from data/mock directory
    logger.info("Indexing policy documents...")
    data_dir = Path("data/mock")
    if data_dir.exists():
        index_stats = rag_pipeline.index_documents(
            data_dir=data_dir,
            clear_existing=False  # Don't clear - skills are already indexed
        )
        logger.info(f"Indexed {index_stats.get('num_documents', 0)} documents, {index_stats.get('num_chunks', 0)} chunks")
    else:
        logger.warning("Policy documents directory not found")
    
    # Initialize Skill Registry
    logger.info("Loading Skill Registry...")
    skill_registry = SkillRegistry(
        embedding_service=rag_pipeline.embedding_service,
        vector_store=rag_pipeline.vector_store
    )
    
    # Load and index skills from YAML files
    logger.info("Loading and indexing skills from YAML files...")
    skill_ids = list(skill_registry._metadata.keys())
    logger.info(f"Found {len(skill_ids)} skills in registry to load and index")
    
    indexed_count = 0
    for skill_id in skill_ids:
        try:
            # Load skill from YAML file
            skill_file = f"data/skills/{skill_id}.yaml"
            skill = skill_registry._load_skill_from_file(skill_file)
            
            if skill:
                logger.info(f"Loaded: {skill.skill_id} ({skill.name}) with {len(skill.steps)} steps")
                
                # Index the skill triggers
                result = skill_registry.index_skill_triggers(skill.skill_id)
                if result:
                    indexed_count += 1
                    logger.info(f"Indexed {skill.skill_id}")
                else:
                    logger.warning(f"Failed to index {skill.skill_id}")
            else:
                logger.error(f"Failed to load {skill_id}")
                
        except Exception as e:
            logger.warning(f"Failed to load/index skill {skill_id}: {e}")
    
    logger.info(f"Indexed {indexed_count}/{len(skill_ids)} skill triggers")
    
    # Initialize Tool Registry
    logger.info("Setting up Tool Registry...")
    tool_registry = ToolRegistry()
    
    # Initialize LLM Service
    logger.info("Initializing LLM Service...")
    llm_service = LLMService()
    
    # Initialize Skill Matcher
    logger.info("Setting up Skill Matcher...")
    skill_matcher = SkillMatcher(
        skill_registry=skill_registry,
        embedding_service=rag_pipeline.embedding_service,
        vector_store=rag_pipeline.vector_store,
        llm_service=llm_service,
        enable_query_reformulation=True
    )
    
    # Initialize ReAct Loop
    logger.info("Initializing ReAct Loop...")
    tpao_loop = TPAOLoop(
        rag_pipeline=rag_pipeline,
        tool_registry=tool_registry,
        llm_service=llm_service,
        max_iterations=5
    )
    
    # Initialize Approval Service
    logger.info("Setting up Approval Service...")
    approval_service = ApprovalService()
    
    # Initialize Executor Agent
    logger.info("Initializing Executor Agent...")
    executor_agent = ExecutorAgent(
        tool_registry=tool_registry,
        approval_service=approval_service
    )
    
    logger.info("System initialization complete!")
    
    return {
        'rag_pipeline': rag_pipeline,
        'skill_registry': skill_registry,
        'tool_registry': tool_registry,
        'llm_service': llm_service,
        'skill_matcher': skill_matcher,
        'tpao_loop': tpao_loop,
        'approval_service': approval_service,
        'executor_agent': executor_agent,
        'indexed_skills': indexed_count,
        'total_skills': len(skill_ids)
    }

def ensure_system_initialized():
    """Ensure system is initialized, auto-initialize if needed."""
    if not st.session_state.initialized:
        try:
            with st.spinner("🚀 Initializing system components..."):
                components = initialize_system_components()
                
                # Store in session state
                st.session_state.rag_pipeline = components['rag_pipeline']
                st.session_state.skill_registry = components['skill_registry']
                st.session_state.tool_registry = components['tool_registry']
                st.session_state.llm_service = components['llm_service']
                st.session_state.skill_matcher = components['skill_matcher']
                st.session_state.tpao_loop = components['tpao_loop']
                st.session_state.approval_service = components['approval_service']
                st.session_state.executor_agent = components['executor_agent']
                st.session_state.initialized = True
                
                st.success(f"✅ System initialized! Loaded {components['indexed_skills']}/{components['total_skills']} skills")
                logger.info("System components stored in session state")
                
        except Exception as e:
            st.error(f"❌ Initialization failed: {str(e)}")
            logger.exception("System initialization failed")
            st.session_state.initialized = False

def render_header():
    """Render the main header."""
    st.markdown('<div class="main-header">🤖 Customer Issue Resolution Copilot</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">A self-learning agent that converts customer issues into reusable skills</div>', unsafe_allow_html=True)

def render_sidebar():
    """Render the sidebar with system status and navigation."""
    with st.sidebar:
        st.image("https://via.placeholder.com/300x100/1f77b4/ffffff?text=Hotel+Copilot", use_container_width=True)
        
        st.markdown("---")
        
        # System Status
        st.subheader("🔧 System Status")
        if st.session_state.initialized:
            st.success("✅ All systems operational")
            
            # Component status
            with st.expander("📊 Component Details"):
                st.write("✅ RAG Pipeline: Active")
                st.write(f"✅ Skills: {len(st.session_state.skill_registry.list_skills()) if st.session_state.skill_registry else 0} loaded")
                st.write(f"✅ Tools: {len(st.session_state.tool_registry.list_tools()) if st.session_state.tool_registry else 0} available")
                st.write("✅ ReAct Loop: Ready")
                st.write("✅ Approval Service: Active")
                st.write("✅ Executor Agent: Ready")
        else:
            st.warning("⚠️ System initializing...")
        
        st.markdown("---")
        
        # Navigation
        st.subheader("📍 Navigation")
        page = st.radio(
            "Select Page",
            ["🏠 Dashboard", "📝 Submit Issue", "✅ Approvals", "📚 Skills", "📖 Knowledge Base", "🛠️ Tools"],
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        
        # Quick Stats
        st.subheader("📈 Quick Stats")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Issues", len(st.session_state.issues))
        with col2:
            st.metric("Resolved", len([r for r in st.session_state.resolutions if r.status == ResolutionStatus.COMPLETED]))
        
        return page

def render_dashboard():
    """Render the main dashboard."""
    st.header("🏠 Dashboard")
    
    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Issues",
            len(st.session_state.issues),
            delta="+2 today" if len(st.session_state.issues) > 0 else None
        )
    
    with col2:
        resolved = len([r for r in st.session_state.resolutions if r.status == ResolutionStatus.COMPLETED])
        st.metric("Resolved", resolved)
    
    with col3:
        pending_approvals = len([a for a in st.session_state.approvals if a.status == ApprovalStatus.PENDING])
        st.metric("Pending Approvals", pending_approvals, delta="Needs attention" if pending_approvals > 0 else None)
    
    with col4:
        skills_count = len(st.session_state.skill_registry.list_skills()) if st.session_state.skill_registry else 0
        st.metric("Available Skills", skills_count)
    
    st.markdown("---")
    
    # Recent Activity
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📋 Recent Issues")
        if st.session_state.issues:
            for issue in st.session_state.issues[-5:]:
                with st.expander(f"🎫 {issue.issue_id}: {issue.subject}"):
                    st.write(f"**Created:** {issue.timestamp.strftime('%Y-%m-%d %H:%M')}")
                    if issue.guest_email:
                        st.write(f"**Guest:** {issue.guest_email}")
                    if issue.booking_id:
                        st.write(f"**Booking:** {issue.booking_id}")
        else:
            st.info("No issues submitted yet. Use 'Submit Issue' to create one.")
    
    with col2:
        st.subheader("✅ Recent Resolutions")
        if st.session_state.resolutions:
            for resolution in st.session_state.resolutions[-5:]:
                status_emoji = "✅" if resolution.status == ResolutionStatus.COMPLETED else "⏳"
                with st.expander(f"{status_emoji} {resolution.resolution_id}"):
                    st.write(f"**Status:** {resolution.status}")
                    st.write(f"**Steps:** {len(resolution.steps)}")
                    if resolution.completed_at:
                        st.write(f"**Completed:** {resolution.completed_at.strftime('%Y-%m-%d %H:%M')}")
                    
                    # Display citations if available (from ReAct Think phase)
                    for step in resolution.steps:
                        if step.step_type == "think" and step.output_data and "citations" in step.output_data:
                            citations = step.output_data["citations"]
                            if citations:
                                st.write("**📚 Policy Citations:**")
                                for citation in citations:
                                    source_name = citation.get("source_name", "Unknown")
                                    doc_type = citation.get("doc_type", "document")
                                    score = citation.get("relevance_score", 0)
                                    preview = citation.get("content_preview", "")[:150]
                                    
                                    # Color code by relevance
                                    score_emoji = "🟢" if score > 0.8 else "🟡" if score > 0.5 else "🔴"
                                    
                                    st.markdown(f"{score_emoji} **{source_name}** ({doc_type}) - Relevance: {score:.2f}")
                                    st.caption(f"_{preview}..._")
        else:
            st.info("No resolutions yet.")

def render_submit_issue():
    """Render the issue submission form."""
    st.header("📝 Submit Customer Issue")
    
    with st.form("issue_form"):
        subject = st.text_input("Subject *", placeholder="Brief description of the issue")
        
        col1, col2 = st.columns(2)
        with col1:
            guest_id = st.text_input("Guest ID", placeholder="Optional: G12345")
        with col2:
            booking_id = st.text_input("Booking ID", placeholder="Optional: BK67890")
        
        description = st.text_area(
            "Description *",
            placeholder="Detailed description of the customer issue...",
            height=150
        )
        
        submitted = st.form_submit_button("🚀 Submit Issue", use_container_width=True)
        
        if submitted:
            if not subject or not description:
                st.error("❌ Please fill in all required fields (marked with *)")
            else:
                try:
                    # ============================================================
                    # GUARDRAIL LAYER 1: PII DETECTION AND MASKING
                    # ============================================================
                    pii_detector = get_pii_detector()
                    
                    # Check subject for PII
                    masked_subject, subject_pii = pii_detector.detect_and_mask(subject)
                    
                    # Check description for PII
                    masked_description, desc_pii = pii_detector.detect_and_mask(description)
                    
                    # Combine all PII findings
                    all_pii = subject_pii + desc_pii
                    
                    # Check if request should be blocked due to excessive PII
                    should_block_pii, pii_reason = pii_detector.should_block_request(description)
                    
                    if should_block_pii:
                        st.error(f"🚫 **Security Alert: Request Blocked**")
                        st.error(f"**Reason:** {pii_reason}")
                        st.warning("⚠️ Your request contains excessive sensitive information. Please remove sensitive data and try again.")
                        return
                    
                    # Show PII masking info if any PII was detected
                    if all_pii:
                        st.warning(f"🔒 **PII Detected and Masked:** {len(all_pii)} sensitive item(s) found")
                        with st.expander("View Masked PII Details"):
                            for pii_match in all_pii:
                                st.write(f"- **{pii_match.pii_type.value}**: {pii_match.masked_value} (confidence: {pii_match.confidence:.2f})")
                    
                    # ============================================================
                    # GUARDRAIL LAYER 2: PROMPT INJECTION DETECTION
                    # ============================================================
                    injection_detector = get_injection_detector()
                    
                    # Check subject for injection attempts
                    subject_injection = injection_detector.check_content(masked_subject)
                    
                    # Check description for injection attempts
                    desc_injection = injection_detector.check_content(masked_description)
                    
                    # Check if content should be blocked
                    should_block_subject, subject_reason = injection_detector.should_block_content(subject_injection)
                    should_block_desc, desc_reason = injection_detector.should_block_content(desc_injection)
                    
                    if should_block_subject or should_block_desc:
                        st.error(f"🚫 **Security Alert: Malicious Content Detected**")
                        
                        if should_block_subject:
                            st.error(f"**Subject:** {subject_reason}")
                            for threat in subject_injection.threats_detected:
                                st.write(f"  - {threat.threat_type.value}: Pattern '{threat.matched_pattern}' (severity: {threat.severity})")
                        
                        if should_block_desc:
                            st.error(f"**Description:** {desc_reason}")
                            for threat in desc_injection.threats_detected:
                                st.write(f"  - {threat.threat_type.value}: Pattern '{threat.matched_pattern}' (severity: {threat.severity})")
                        
                        st.warning("⚠️ Your request appears to contain malicious patterns. Please rephrase and try again.")
                        logger.warning(f"Blocked injection attempt - Subject threats: {len(subject_injection.threats_detected)}, Desc threats: {len(desc_injection.threats_detected)}")
                        return
                    
                    # Log any low-severity threats that were detected but not blocked
                    if subject_injection.threats_detected or desc_injection.threats_detected:
                        total_threats = len(subject_injection.threats_detected) + len(desc_injection.threats_detected)
                        logger.info(f"Low-severity threats detected but allowed: {total_threats}")
                    
                    # ============================================================
                    # CREATE ISSUE WITH MASKED CONTENT
                    # ============================================================
                    # Create issue with default values for removed fields
                    issue = Issue(
                        issue_id=f"ISS-{len(st.session_state.issues) + 1:04d}",
                        channel="email",  # Default channel
                        subject=masked_subject,  # Use masked subject
                        body=masked_description,  # Use masked description
                        issue_type="other",  # Default type (valid enum value)
                        priority="medium",  # Default priority
                        guest_email=guest_id if guest_id else None,
                        booking_id=booking_id if booking_id else None,
                        metadata={
                            "pii_detected": len(all_pii) > 0,
                            "pii_count": len(all_pii),
                            "injection_threats_detected": len(subject_injection.threats_detected) + len(desc_injection.threats_detected)
                        }
                    )
                    
                    st.session_state.issues.append(issue)
                    
                    st.success(f"✅ Issue {issue.issue_id} created successfully!")
                    
                    # Show next steps
                    st.info("🔄 The system will now:\n"
                           "1. Check for matching skills\n"
                           "2. Retrieve relevant knowledge\n"
                           "3. Generate resolution plan\n"
                           "4. Request approval if needed")
                    
                    # Auto-process using actual skill matching
                    with st.spinner("🧠 Processing issue..."):
                        # Step 1: Try to match existing skill using the initialized skill matcher
                        matches = st.session_state.skill_matcher.match_skill(issue, top_k=3)
                        
                        # Display all matches for transparency
                        if matches:
                            st.write("**Skill Matching Results:**")
                            for i, match in enumerate(matches, 1):
                                confidence_emoji = "🟢" if match.confidence == "high" else "🟡" if match.confidence == "medium" else "🔴"
                                st.write(f"{i}. {match.skill.name}: {confidence_emoji} {match.confidence} (score: {match.score:.3f})")
                            
                            # NEW APPROACH: Always use top matched skill (regardless of confidence)
                            # Human will approve/reject in the Approvals page
                            matched_skill = matches[0].skill
                            st.success(f"✅ Matched skill: **{matched_skill.name}** (will be sent for approval)")
                            
                            # Create resolution from skill steps
                            resolution_steps = []
                            for i, skill_step in enumerate(matched_skill.steps, 1):
                                resolution_steps.append(
                                    ResolutionStep(
                                        step_number=i,
                                        step_type=skill_step.step_type.value if hasattr(skill_step.step_type, 'value') else "tool_call",
                                        description=skill_step.description,
                                        tool_used=skill_step.tool_name,
                                        input_data=skill_step.parameters,
                                        output_data={},
                                        success=not skill_step.requires_approval,  # Pending if requires approval
                                        duration_ms=0.0  # Will be updated during execution
                                    )
                                )
                            
                            resolution = Resolution(
                                resolution_id=f"RES-{issue.issue_id}",
                                issue_id=issue.issue_id,
                                status=ResolutionStatus.PENDING_APPROVAL,  # Always pending approval
                                steps=resolution_steps,
                                skill_used=matched_skill.skill_id,
                                skill_matched=True,
                                novel_task=False,
                                approval_request_id=None,  # Will be set when approval is created
                                approval_granted=None,
                                outcome=None,
                                guest_satisfaction=None,
                                completed_at=None,
                                total_duration_ms=None,
                                compiled_skill_id=None
                            )
                        else:
                            # No skill match at all - go directly to ReAct
                            st.write("**Skill Matching Results:**")
                            st.write("No skills matched the query.")
                            st.warning("⚠️ No matching skill found. Using ReAct loop for novel task handling...")
                            
                            # Use ReAct loop to generate plan - it returns a Resolution object
                            resolution = st.session_state.tpao_loop.execute(issue)
                            
                            # Update resolution ID and status for consistency
                            resolution.resolution_id = f"RES-{issue.issue_id}"
                            resolution.status = ResolutionStatus.PENDING_APPROVAL
                        
                        st.session_state.resolutions.append(resolution)
                        
                        # Create approval request for actions requiring approval
                        # Build actions list with more detail
                        actions_list = []
                        for step in resolution.steps:
                            action_detail = {
                                "step": step.step_number,
                                "type": step.step_type,
                                "description": step.description,
                                "tool": step.tool_used if step.tool_used else "N/A"
                            }
                            actions_list.append(action_detail)
                        
                        approval = ApprovalRequest(
                            issue_id=issue.issue_id,
                            action_type="skill_execution" if resolution.skill_used else "novel_task_resolution",
                            action_description=f"Execute {'skill: ' + resolution.skill_used if resolution.skill_used else 'novel task resolution'} for issue: {masked_subject}",
                            risk_level=RiskLevel.MEDIUM,  # Default medium risk
                            proposed_changes={
                                "resolution_id": resolution.resolution_id,
                                "skill_used": resolution.skill_used if hasattr(resolution, 'skill_used') and resolution.skill_used else "ReAct",
                                "total_steps": len(resolution.steps),
                                "actions": actions_list,
                                "match_score": matches[0].score if matches else None,
                                "match_confidence": matches[0].confidence if matches else None
                            }
                        )
                        
                        # Update resolution with approval request ID
                        resolution.approval_request_id = approval.request_id
                        
                        # Add to session state for UI display
                        st.session_state.approvals.append(approval)
                        
                        # Register with ApprovalService so it can be approved/rejected
                        st.session_state.approval_service.approval_history.append(approval)
                    
                    st.success(f"✅ Resolution plan created: {resolution.resolution_id}")
                    st.info(f"⏳ Approval request created: {approval.request_id}")
                    
                except Exception as e:
                    st.error(f"❌ Error creating issue: {str(e)}")

def render_approvals():
    """Render the approvals page."""
    st.header("✅ Approval Requests")
    
    # Filter
    status_filter = st.selectbox(
        "Filter by Status",
        options=["All", "Pending", "Approved", "Rejected"]
    )
    
    # Get filtered approvals
    approvals = st.session_state.approvals
    if status_filter != "All":
        approvals = [a for a in approvals if a.status.value == status_filter.lower()]
    
    if not approvals:
        st.info("No approval requests found.")
        return
    
    # Display approvals
    for approval in reversed(approvals):
        risk_color = {
            RiskLevel.LOW: "🟢",
            RiskLevel.MEDIUM: "🟡",
            RiskLevel.HIGH: "🔴",
            RiskLevel.CRITICAL: "🔴"
        }.get(approval.risk_level, "⚪")
        
        status_emoji = {
            ApprovalStatus.PENDING: "⏳",
            ApprovalStatus.APPROVED: "✅",
            ApprovalStatus.REJECTED: "❌"
        }.get(approval.status, "⚪")
        
        with st.expander(f"{status_emoji} {approval.request_id} - {risk_color} {approval.risk_level.value.upper()}"):
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.write(f"**Issue ID:** {approval.issue_id}")
                st.write(f"**Action:** {approval.action_description}")
                st.write(f"**Risk Level:** {approval.risk_level.value.upper()}")
                st.write(f"**Status:** {approval.status.value.upper()}")
                st.write(f"**Created:** {approval.created_at.strftime('%Y-%m-%d %H:%M')}")
                
                # Display policy citations if this is a ReAct resolution
                if approval.action_type in ["tpao_novel_task_resolution", "novel_task_resolution"]:
                    # Find the resolution for this approval
                    resolution = None
                    for res in st.session_state.resolutions:
                        if res.issue_id == approval.issue_id and res.novel_task:
                            resolution = res
                            break
                    
                    if resolution:
                        # Extract citations from Think step
                        for step in resolution.steps:
                            if step.step_type == "think" and step.output_data and "citations" in step.output_data:
                                citations = step.output_data["citations"]
                                if citations:
                                    st.write("---")
                                    st.write("**📚 Policy Citations Used:**")
                                    for citation in citations:
                                        source_name = citation.get("source_name", "Unknown")
                                        doc_type = citation.get("doc_type", "document")
                                        score = citation.get("relevance_score", 0)
                                        preview = citation.get("content_preview", "")[:200]
                                        
                                        # Color code by relevance
                                        if score > 0.8:
                                            score_emoji = "🟢"
                                            score_label = "High"
                                        elif score > 0.5:
                                            score_emoji = "🟡"
                                            score_label = "Medium"
                                        else:
                                            score_emoji = "🔴"
                                            score_label = "Low"
                                        
                                        st.markdown(f"{score_emoji} **{source_name}** ({doc_type})")
                                        st.caption(f"Relevance: {score_label} ({score:.3f})")
                                        with st.expander("📄 View Policy Excerpt"):
                                            st.text(preview + "...")
                                    break
                
                if approval.proposed_changes:
                    st.write("---")
                    st.write("**Proposed Changes:**")
                    st.json(approval.proposed_changes)
            
            with col2:
                if approval.status == ApprovalStatus.PENDING:
                    st.write("**Actions:**")
                    
                    if st.button("✅ Approve", key=f"approve_{approval.request_id}", use_container_width=True):
                        try:
                            token = st.session_state.approval_service.approve_request(
                                request_id=approval.request_id,
                                approver="manager@hotel.com",
                                comments="Approved via UI"
                            )
                            approval.status = ApprovalStatus.APPROVED
                            
                            # Check if this is a ReAct-generated resolution that needs to be compiled into a skill
                            st.write(f"DEBUG: Checking action_type: {approval.action_type}")
                            if approval.action_type in ["tpao_novel_task_resolution", "novel_task_resolution"]:
                                st.info("🔨 Compiling ReAct resolution into reusable skill...")
                                
                                # Find the resolution for this approval using issue_id and novel_task flag
                                resolution = None
                                st.write(f"DEBUG: Looking for resolution with issue_id: {approval.issue_id}")
                                st.write(f"DEBUG: Total resolutions in state: {len(st.session_state.resolutions)}")
                                
                                for res in st.session_state.resolutions:
                                    st.write(f"DEBUG: Checking resolution {res.resolution_id}, issue_id={res.issue_id}, novel_task={res.novel_task}")
                                    if res.issue_id == approval.issue_id and res.novel_task:
                                        resolution = res
                                        st.write(f"DEBUG: Found matching ReAct resolution!")
                                        break
                                
                                if resolution:
                                    st.write(f"DEBUG: Resolution found with {len(resolution.steps)} steps")
                                    # Find the original issue
                                    issue = None
                                    st.write(f"DEBUG: Looking for issue_id: {approval.issue_id}")
                                    for iss in st.session_state.issues:
                                        if iss.issue_id == approval.issue_id:
                                            issue = iss
                                            st.write(f"DEBUG: Found matching issue!")
                                            break
                                    
                                    if issue:
                                        st.write("DEBUG: Starting skill compilation...")
                                        try:
                                            # Create a new skill from the ReAct resolution
                                            from src.domain.models.skill import Skill, SkillStep, SkillStepType, SkillStatus
                                            import secrets
                                            import yaml
                                            from pathlib import Path
                                            
                                            # Generate skill ID and name
                                            skill_id = f"skill_{secrets.token_hex(4)}"
                                            
                                            # Create descriptive skill name from issue subject
                                            # Clean up the subject and make it a proper skill name
                                            subject_words = issue.subject.strip().split()[:5]  # Max 5 words
                                            skill_name = ' '.join(subject_words).title()
                                            
                                            # Add "Handler" suffix if not present
                                            if not skill_name.lower().endswith('handler'):
                                                skill_name = f"{skill_name} Handler"
                                            
                                            # Fallback to issue type if subject is too short
                                            if len(skill_name) < 10:
                                                skill_name = f"{issue.issue_type.value.replace('_', ' ').title()} Handler"
                                            
                                            st.write(f"✨ Skill name: **{skill_name}**")
                                            
                                            # Generate semantic trigger variations using LLM
                                            st.write("🔄 Generating semantic trigger variations...")
                                            trigger_variations = []
                                            try:
                                                trigger_prompt = f"""Generate 5-7 semantic variations of this customer request for skill matching.

Original request: "{issue.subject or issue.body[:100]}"

Generate variations that capture the same intent but use different phrasings. Include:
1. Formal versions
2. Casual versions
3. Different word choices
4. Common synonyms
5. Related phrases

Return ONLY a JSON array of strings, no explanation:
["variation 1", "variation 2", ...]

Examples:
Original: "i need to checkin early"
Variations: ["early check-in request", "early access to room", "check in before standard time", "arrive early at hotel", "can I check in early", "early arrival request", "need room access before 3pm"]
"""
                                                
                                                llm_response = st.session_state.llm_service.generate_json_simple(trigger_prompt)
                                                
                                                if isinstance(llm_response, list) and len(llm_response) > 0:
                                                    trigger_variations = llm_response[:7]  # Max 7 variations
                                                    st.write(f"✅ Generated {len(trigger_variations)} trigger variations")
                                                    # Show first 3 as preview
                                                    for i, trigger in enumerate(trigger_variations[:3], 1):
                                                        st.write(f"  {i}. {trigger}")
                                                    if len(trigger_variations) > 3:
                                                        st.write(f"  ... and {len(trigger_variations) - 3} more")
                                                else:
                                                    st.warning("⚠️ LLM returned invalid format, using fallback triggers")
                                                    trigger_variations = []
                                            except Exception as e:
                                                st.warning(f"⚠️ Trigger generation failed: {e}, using fallback")
                                                trigger_variations = []
                                            
                                            # Combine original + variations, remove duplicates
                                            all_triggers = [issue.subject or issue.body[:100]]
                                            if trigger_variations:
                                                all_triggers.extend(trigger_variations)
                                            
                                            # Remove duplicates while preserving order
                                            seen = set()
                                            unique_triggers = []
                                            for trigger in all_triggers:
                                                trigger_lower = trigger.lower().strip()
                                                if trigger_lower and trigger_lower not in seen:
                                                    seen.add(trigger_lower)
                                                    unique_triggers.append(trigger)
                                            
                                            # Limit to 8 triggers max
                                            final_triggers = unique_triggers[:8]
                                            st.write(f"📝 Using {len(final_triggers)} triggers for skill matching")
                                            
                                            # Convert resolution steps to skill steps
                                            skill_steps = []
                                            for idx, res_step in enumerate(resolution.steps):
                                                step_type = SkillStepType.TOOL_CALL if res_step.tool_used else SkillStepType.REASONING
                                                
                                                # Convert output_data to string if it's a dict
                                                expected_output_str = None
                                                if res_step.output_data:
                                                    if isinstance(res_step.output_data, dict):
                                                        import json
                                                        expected_output_str = json.dumps(res_step.output_data)
                                                    else:
                                                        expected_output_str = str(res_step.output_data)
                                                
                                                skill_step = SkillStep(
                                                    step_id=f"step_{idx + 1}",
                                                    step_type=step_type,
                                                    description=res_step.description,
                                                    tool_name=res_step.tool_used,
                                                    parameters=res_step.input_data or {},
                                                    expected_output=expected_output_str,
                                                    requires_approval=not res_step.success,
                                                    approval_reason="Action requires verification" if not res_step.success else None
                                                )
                                                skill_steps.append(skill_step)
                                            
                                            # Check if resolution was created without policy grounding
                                            no_policy_match = resolution.metadata.get("no_policy_match", False)
                                            using_general_knowledge = resolution.metadata.get("using_general_knowledge", False)
                                            
                                            # Add warning to description if no policies were found
                                            skill_description = f"Automatically generated skill for: {issue.subject or issue.body[:50]}"
                                            if no_policy_match:
                                                skill_description = f"⚠️ WARNING: Created without company policy grounding. {skill_description}"
                                            
                                            # Create the skill with generated trigger variations
                                            new_skill = Skill(
                                                skill_id=skill_id,
                                                name=skill_name,
                                                description=skill_description,
                                                version="1.0",
                                                triggers=final_triggers,  # Use LLM-generated semantic variations
                                                steps=skill_steps,
                                                status=SkillStatus.ACTIVE,
                                                metadata={
                                                    "created_from": "tpao_trace",
                                                    "source_issue_id": issue.issue_id,
                                                    "source_resolution_id": resolution.resolution_id,
                                                    "domain": "hotel_operations",
                                                    "no_policy_match": no_policy_match,
                                                    "using_general_knowledge": using_general_knowledge,
                                                    "requires_policy_review": no_policy_match  # Flag for human review
                                                },
                                                created_by="tpao_compiler"
                                            )
                                            
                                            # Save skill to YAML file
                                            skills_dir = Path("data/skills")
                                            skills_dir.mkdir(parents=True, exist_ok=True)
                                            skill_file = skills_dir / f"{skill_id}.yaml"
                                            
                                            # Convert skill to YAML format
                                            skill_dict = {
                                                "skill_id": new_skill.skill_id,
                                                "name": new_skill.name,
                                                "version": new_skill.version,
                                                "description": new_skill.description,
                                                "triggers": new_skill.triggers,
                                                "steps": [
                                                    {
                                                        "step_id": step.step_id,
                                                        "step_type": step.step_type.value,
                                                        "description": step.description,
                                                        "tool_name": step.tool_name,
                                                        "parameters": step.parameters,
                                                        "requires_approval": step.requires_approval,
                                                        "approval_reason": step.approval_reason,
                                                        "expected_output": step.expected_output
                                                    }
                                                    for step in new_skill.steps
                                                ],
                                                "guardrails": new_skill.guardrails,
                                                "metadata": new_skill.metadata
                                            }
                                            
                                            with open(skill_file, 'w') as f:
                                                yaml.dump(skill_dict, f, default_flow_style=False, sort_keys=False)
                                            
                                            # Show success message with warning if applicable
                                            if no_policy_match:
                                                st.warning(f"⚠️ Skill created but requires policy review - no company policies were found during resolution")
                                                st.info(f"📝 Skill saved to: {skill_file}")
                                            else:
                                                st.success(f"✅ Skill YAML saved to: {skill_file}")
                                            
                                            # Add to registry metadata and index in vector DB
                                            from src.application.skills.skill_registry import SkillMetadata
                                            from datetime import datetime
                                            
                                            # Create metadata entry
                                            metadata = SkillMetadata(
                                                skill_id=skill_id,
                                                name=skill_name,
                                                file_path=str(skill_file),
                                                domain=new_skill.metadata.get("domain", "hotel_operations"),
                                                category=new_skill.metadata.get("category", "general"),
                                                active=True,
                                                version=new_skill.version,
                                                created_at=datetime.now().isoformat(),
                                                usage_count=0,
                                                success_rate=0.0
                                            )
                                            
                                            # Add to registry's internal metadata
                                            st.session_state.skill_registry._metadata[skill_id] = metadata
                                            st.session_state.skill_registry._skills[skill_id] = new_skill
                                            
                                            # Save registry file
                                            st.session_state.skill_registry._save_registry()
                                            
                                            # Index the skill triggers in vector DB for matching
                                            st.write("DEBUG: Indexing skill triggers...")
                                            try:
                                                # Generate embedding for enriched trigger text
                                                enriched_trigger_text = (
                                                    f"Skill: {new_skill.name}. "
                                                    f"Description: {new_skill.description}. "
                                                    f"Triggers: {' '.join(new_skill.triggers)}"
                                                )
                                                
                                                embedding_result = st.session_state.skill_registry.embedding_service.embed_texts([enriched_trigger_text])
                                                embedding = embedding_result.embeddings[0]
                                                
                                                # Serialize skill data to JSON
                                                import json
                                                skill_data_json = json.dumps({
                                                    "skill_id": new_skill.skill_id,
                                                    "version": new_skill.version,
                                                    "name": new_skill.name,
                                                    "description": new_skill.description,
                                                    "triggers": new_skill.triggers,
                                                    "steps": [
                                                        {
                                                            "step_id": step.step_id,
                                                            "step_type": step.step_type.value,
                                                            "description": step.description,
                                                            "tool_name": step.tool_name,
                                                            "parameters": step.parameters or {},
                                                            "requires_approval": step.requires_approval,
                                                            "approval_reason": step.approval_reason,
                                                            "expected_output": step.expected_output
                                                        }
                                                        for step in new_skill.steps
                                                    ],
                                                    "metadata": new_skill.metadata or {},
                                                    "guardrails": new_skill.guardrails or {}
                                                })
                                                
                                                # Add to vector DB
                                                doc_id = f"skill_{skill_id}"
                                                st.session_state.skill_registry.vector_store.add_documents(
                                                    chunk_ids=[doc_id],
                                                    contents=[enriched_trigger_text],
                                                    embeddings=[embedding],
                                                    metadatas=[{
                                                        "doc_type": "skill_trigger",
                                                        "skill_id": skill_id,
                                                        "domain": new_skill.metadata.get('domain', ''),
                                                        "active": True,
                                                        "skill_data_json": skill_data_json
                                                    }]
                                                )
                                                
                                                st.success(f"✅ **New Skill Learned:** {new_skill.name}")
                                                st.info(f"📄 File: {skill_file}")
                                                st.info(f"🔢 Skill ID: {skill_id}")
                                                st.info("♻️ Similar issues will now match this skill automatically")
                                                
                                                # Force UI refresh to show updated skill count in sidebar
                                                import time
                                                time.sleep(1)
                                                st.rerun()
                                            except Exception as index_error:
                                                st.warning(f"⚠️ Skill created but indexing failed: {str(index_error)}")
                                                import traceback
                                                st.code(traceback.format_exc())
                                                
                                        except Exception as compile_error:
                                            st.error(f"❌ Skill compilation failed: {str(compile_error)}")
                                            import traceback
                                            st.code(traceback.format_exc())
                                    else:
                                        st.error("DEBUG: Issue not found - cannot compile skill")
                                else:
                                    st.error("DEBUG: No ReAct resolution found for this issue")
                            else:
                                st.write(f"DEBUG: Action type '{approval.action_type}' does not match ReAct types")
                                # For non-ReAct approvals, show success and rerun
                                st.success(f"✅ Approved! Token: {token.token[:16]}...")
                                st.rerun()
                        except Exception as e:
                            st.error(f"❌ Error: {str(e)}")
                            import traceback
                            st.error(traceback.format_exc())
                    
                    if st.button("❌ Reject & Use ReAct", key=f"reject_{approval.request_id}", use_container_width=True):
                        try:
                            # Reject the approval
                            st.session_state.approval_service.reject_request(
                                request_id=approval.request_id,
                                approver="manager@hotel.com",
                                reason="Skill rejected - Using ReAct for novel task handling"
                            )
                            approval.status = ApprovalStatus.REJECTED
                            
                            # Find the original issue for this approval
                            issue = None
                            for iss in st.session_state.issues:
                                if iss.issue_id == approval.issue_id:
                                    issue = iss
                                    break
                            
                            if issue:
                                st.info("🔄 Skill rejected. Triggering ReAct loop for novel task handling...")
                                
                                # Use ReAct loop to generate new plan
                                with st.spinner("🧠 ReAct loop generating new resolution plan..."):
                                    tpao_resolution = st.session_state.tpao_loop.execute(issue)
                                    
                                    # Update resolution ID and mark as novel task
                                    tpao_resolution.resolution_id = f"RES-{issue.issue_id}-ReAct"
                                    tpao_resolution.status = ResolutionStatus.PENDING_APPROVAL
                                    tpao_resolution.novel_task = True
                                    tpao_resolution.skill_matched = False
                                    
                                    # Add to resolutions
                                    st.session_state.resolutions.append(tpao_resolution)
                                    
                                    # Create new approval request for ReAct resolution
                                    actions_list = []
                                    for step in tpao_resolution.steps:
                                        action_detail = {
                                            "step": step.step_number,
                                            "type": step.step_type,
                                            "description": step.description,
                                            "tool": step.tool_used if step.tool_used else "N/A"
                                        }
                                        actions_list.append(action_detail)
                                    
                                    tpao_approval = ApprovalRequest(
                                        issue_id=issue.issue_id,
                                        action_type="tpao_novel_task_resolution",
                                        action_description=f"ReAct-generated resolution for: {issue.subject}",
                                        risk_level=RiskLevel.MEDIUM,
                                        proposed_changes={
                                            "resolution_id": tpao_resolution.resolution_id,
                                            "skill_used": "ReAct (Novel Task)",
                                            "total_steps": len(tpao_resolution.steps),
                                            "actions": actions_list,
                                            "triggered_by": "skill_rejection"
                                        }
                                    )
                                    
                                    # Update resolution with approval request ID
                                    tpao_resolution.approval_request_id = tpao_approval.request_id
                                    
                                    # Add to session state
                                    st.session_state.approvals.append(tpao_approval)
                                    st.session_state.approval_service.approval_history.append(tpao_approval)
                                    
                                    st.success(f"✅ ReAct resolution created: {tpao_resolution.resolution_id}")
                                    st.info(f"⏳ New approval request created: {tpao_approval.request_id}")
                            else:
                                st.warning("⚠️ Could not find original issue to trigger ReAct")
                            
                            st.rerun()
                        except Exception as e:
                            st.error(f"❌ Error: {str(e)}")
                            import traceback
                            st.error(traceback.format_exc())
                else:
                    if approval.approved_by:
                        st.write(f"**By:** {approval.approved_by}")
                    if approval.approved_at:
                        st.write(f"**At:** {approval.approved_at.strftime('%Y-%m-%d %H:%M')}")

def render_skills():
    """Render the skills page."""
    st.header("📚 Available Skills")
    
    if not st.session_state.skill_registry:
        st.warning("Skill registry not initialized")
        return
    
    # Use get_all_skills() to get full Skill objects, not just metadata
    skills = st.session_state.skill_registry.get_all_skills(active_only=False)
    
    if not skills:
        st.info("No skills loaded yet.")
        return
    
    st.write(f"**Total Skills:** {len(skills)}")
    
    for skill in skills:
        with st.expander(f"🎯 {skill.name}"):
            st.write(f"**ID:** {skill.skill_id}")
            st.write(f"**Version:** {skill.version}")
            st.write(f"**Status:** {skill.status.value}")
            st.write(f"**Description:** {skill.description}")
            st.write(f"**Usage Count:** {skill.usage_count}")
            st.write(f"**Success Rate:** {skill.success_rate:.1%}")
            
            if skill.triggers:
                st.write("**Triggers:**")
                for trigger in skill.triggers:
                    st.write(f"- {trigger}")
            
            if skill.steps:
                st.write(f"**Steps:** {len(skill.steps)}")
                for i, step in enumerate(skill.steps, 1):
                    st.write(f"{i}. {step.description}")

def render_knowledge_base():
    """Render the knowledge base page showing loaded policies."""
    st.header("📖 Knowledge Base")
    
    if not st.session_state.rag_pipeline:
        st.warning("RAG pipeline not initialized")
        return
    
    st.write("This page shows all policy documents loaded into the knowledge base.")
    
    # Get policy files from data/mock/policies directory
    policies_dir = Path("data/mock/policies")
    
    if not policies_dir.exists():
        st.error("Policies directory not found")
        return
    
    policy_files = list(policies_dir.glob("*.md"))
    
    if not policy_files:
        st.info("No policy documents found.")
        return
    
    st.write(f"**Total Policies Loaded:** {len(policy_files)}")
    st.markdown("---")
    
    # Display each policy in an expander
    for policy_file in sorted(policy_files):
        policy_name = policy_file.stem.replace('_', ' ').title()
        
        with st.expander(f"📄 {policy_name}"):
            try:
                with open(policy_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Show file metadata
                st.write(f"**File:** `{policy_file.name}`")
                st.write(f"**Size:** {len(content)} characters")
                
                # Show content preview
                st.markdown("**Content:**")
                st.markdown(content)
                
            except Exception as e:
                st.error(f"Error reading policy: {e}")
    
    # Show vector store statistics
    st.markdown("---")
    st.subheader("📊 Vector Store Statistics")
    
    try:
        # Get collection info from vector store
        collection = st.session_state.rag_pipeline.vector_store.collection
        count = collection.count()
        
        st.metric("Total Chunks Indexed", count)
        st.info("💡 These policies are used by the RAG system to ground agent responses and provide accurate information.")
        
    except Exception as e:
        st.warning(f"Could not retrieve vector store statistics: {e}")

def render_tools():
    """Render the tools page."""
    st.header("🛠️ Available Tools")
    
    if not st.session_state.tool_registry:
        st.warning("Tool registry not initialized")
        return
    
    tools = st.session_state.tool_registry.list_tools()
    
    if not tools:
        st.info("No tools registered yet.")
        return
    
    st.write(f"**Total Tools:** {len(tools)}")
    
    for tool_name in tools:
        tool = st.session_state.tool_registry.get_tool(tool_name)
        if tool:
            with st.expander(f"🔧 {tool_name}"):
                st.write(f"**Name:** {tool.name}")
                st.write(f"**Description:** {tool.description}")
                st.write(f"**Requires Approval:** {tool.requires_approval}")

def main():
    """Main application entry point."""
    # Auto-initialize system on app startup
    ensure_system_initialized()
    
    render_header()
    
    # Render sidebar and get selected page
    page = render_sidebar()
    
    # Render selected page (all pages now accessible after auto-init)
    if not st.session_state.initialized:
        st.warning("⚠️ System is initializing, please wait...")
        return
    
    if page == "🏠 Dashboard":
        render_dashboard()
    elif page == "📝 Submit Issue":
        render_submit_issue()
    elif page == "✅ Approvals":
        render_approvals()
    elif page == "📚 Skills":
        render_skills()
    elif page == "📖 Knowledge Base":
        render_knowledge_base()
    elif page == "🛠️ Tools":
        render_tools()

if __name__ == "__main__":
    main()
