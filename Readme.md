# Customer Issue Resolution Copilot

> **A self-learning agent that turns scattered company knowledge into executable, human-approved skills where every novel task it handles becomes a reusable skill it never has to relearn**

## Vision: A Generic Troubleshooting Framework

This project is designed as a **domain-agnostic customer issue resolution copilot** that can be adapted to any operational context. The core innovation is a system that:

1. **Learns from every resolution** - Novel issues become reusable skills
2. **Grounds responses in company knowledge** - Uses RAG to retrieve relevant policies, procedures, and historical resolutions
3. **Requires human approval** - Maintains safety through approval gates for consequential actions
4. **Executes through an autonomous agent** - Skills are executed by an Executor Agent (Phase 2)

### Current Implementation: Hotel Customer Service POC

For rapid prototyping and demonstration, this POC uses **hotel operations data** to showcase the system's capabilities. However, the architecture is intentionally generic and can be adapted to:

- **IT Operations** - Troubleshooting infrastructure issues, incident response, runbook automation
- **Healthcare** - Patient issue resolution, appointment scheduling, insurance claims
- **IBM DataStage Projects** - ETL pipeline troubleshooting, job failure analysis, performance optimization
- **DevOps** - CI/CD pipeline failures, deployment issues, configuration management
- **Customer Support** - Product issues, billing disputes, account management
- **HR Operations** - Onboarding, benefits administration, policy questions

### Intended Use Case: Software Product Integration and Troubleshooting Copilot

The original motivation for this project was to create a copilot for **software product integration and troubleshooting**, where:
- **Knowledge Base**: Product documentation, API references, error catalogs, integration guides, troubleshooting runbooks
- **Issues**: Integration failures, configuration errors, performance bottlenecks, data pipeline issues, connectivity problems
- **Skills**: Automated diagnostics, log analysis, configuration validation, remediation procedures, restart workflows
- **Executor Agent**: Executes approved remediation actions (restart services, update configs, rollback changes, notify teams)

Examples include ETL tools (IBM DataStage, Informatica), middleware platforms (MuleSoft, Apache Kafka), databases (Oracle, PostgreSQL), and cloud services (AWS, Azure, GCP).

The hotel POC demonstrates the same patterns with more accessible domain knowledge, making it easier to understand and evaluate the system's capabilities.

---

## Architecture Overview

### Core Components

```
┌─────────────────────────────────────────────────────────────┐
│                    Customer Issue Input                      │
│              (Email, Slack, Chat, API, etc.)                 │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                   Orchestrator Agent                         │
│  • Issue Classification                                      │
│  • RAG Retrieval (Policies, Procedures, History)            │
│  • Skill Matching (Check for existing solution)             │
└────────────┬────────────────────────────┬───────────────────┘
             │                            │
    ┌────────▼────────┐          ┌───────▼──────────┐
    │  Skill Path     │          │  Novel Task Path │
    │  (Reuse)        │          │  (ReAct Loop)    │
    └────────┬────────┘          └───────┬──────────┘
             │                            │
             │                    ┌───────▼──────────┐
             │                    │  Think → Plan    │
             │                    │  Act → Observe   │
             │                    └───────┬──────────┘
             │                            │
             └────────────┬───────────────┘
                          │
                          ▼
             ┌────────────────────────┐
             │  Human Approval Layer  │
             │  (Risk Assessment)     │
             └────────────┬───────────┘
                          │
                          ▼
             ┌────────────────────────┐
             │   Executor Agent       │
             │   (Phase 2 - Future)   │
             └────────────┬───────────┘
                          │
                          ▼
             ┌────────────────────────┐
             │  Skill Compilation     │
             │  (Learn for Reuse)     │
             └────────────────────────┘
```

### Key Innovation: Self-Learning Skills System

**The Problem**: Traditional support systems solve the same issues repeatedly, with quality depending on individual operator knowledge.

**The Solution**: Every successfully resolved novel issue becomes a reusable skill:

1. **Novel Issue** → No existing skill matches
2. **ReAct Loop** → Agent reasons through the problem using RAG-retrieved knowledge
3. **Human Approval** → Operator reviews and approves proposed actions
4. **Execution** → Actions are executed (currently manual, Phase 2 will be autonomous)
5. **Skill Compilation** → Successful resolution is compiled into a YAML skill
6. **Future Reuse** → Similar issues automatically match the new skill

---

## Quick Start

### Prerequisites

- Python 3.11+
- OpenAI API key (for embeddings and LLM)
- Cohere API key (optional, for reranking)

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd customer-issue-resolution-copilot

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env and add your API keys
```

### Initialize the System

```bash
# Index hotel policies and knowledge base
python scripts/reindex_hotel_knowledge.py

# Index seed skills
python scripts/reindex_skills_simple.py
```

### Run the Application

```bash
# Start the Streamlit UI
streamlit run app.py
```

Navigate to `http://localhost:8501` to access the interface.

---

## Features

### Implemented (POC)

- **RAG Pipeline**: Semantic search over policies, procedures, and historical resolutions
- **Skill Matching**: Embedding-based matching of issues to existing skills
- **ReAct Loop**: Think-Plan-Act-Observe reasoning for novel tasks
- **Human Approval**: Approval gates for financial, operational, and booking actions
- **Skill Compilation**: Automatic generation of reusable skills from successful resolutions
- **Guardrails**: PII detection, prompt injection resistance, confidence checking
- **Evaluation Framework**: RAG metrics (faithfulness, relevancy, precision, recall)
- **Citation Tracking**: Source attribution for all retrieved knowledge

### In Progress

- **Executor Agent (Phase 2)**: Autonomous execution of approved actions
- **Multi-Agent Orchestration**: Specialized agents for different domains
- **Advanced Error Handling**: Retry logic, rollback mechanisms, escalation workflows

### Future Enhancements

- **Domain Adapters**: Pre-built configurations for IT, Healthcare, DevOps, etc.
- **Real-time Learning**: Continuous skill improvement based on feedback
- **Multi-modal Support**: Handle images, logs, screenshots in issue descriptions
- **Integration Hub**: Connectors for Jira, ServiceNow, Slack, PagerDuty, etc.

---

## Project Structure

```
customer-issue-resolution-copilot/
├── src/                          # Source code
│   ├── domain/                   # Domain models and interfaces
│   ├── application/              # Business logic
│   │   ├── orchestrator/         # Main orchestration agent
│   │   ├── react/                # ReAct reasoning loop
│   │   ├── skills/               # Skill matching and compilation
│   │   ├── rag/                  # RAG pipeline
│   │   ├── guardrails/           # Safety mechanisms
│   │   └── tools/                # Tool implementations
│   └── infrastructure/           # External integrations
├── data/                         # Data storage
│   ├── mock/                     # Mock hotel data (POC)
│   ├── skills/                   # Skill registry (YAML)
│   └── vector_store/             # ChromaDB vector database
├── evals/                        # Evaluation framework
├── tests/                        # Unit and integration tests
├── scripts/                      # Utility scripts
├── docs/                         # Documentation
└── app.py                        # Streamlit UI
```

---

## Adapting to Your Domain

To adapt this system to your own use case (e.g., IBM DataStage, IT operations, healthcare):

### 1. Replace Knowledge Base

```bash
# Add your domain-specific documents to data/mock/policies/
# Examples:
# - DataStage job documentation
# - Error code reference guides
# - Troubleshooting runbooks
# - Historical incident reports
```

### 2. Update Domain Models

Edit `src/domain/models/` to reflect your domain entities:
- Replace `Booking` with your primary entity (e.g., `DataStageJob`, `Ticket`, `Patient`)
- Update `Issue` model with domain-specific fields
- Customize `Context` for your use case

### 3. Implement Domain Tools

Create tools in `src/application/tools/mock_tools.py`:
- Replace hotel tools (lookup_booking, process_refund) with your tools
- Examples: restart_job, check_logs, update_config, notify_oncall

### 4. Seed Initial Skills

Create 3-5 seed skills in `data/skills/` for common issues:
- Use the existing YAML schema
- Define triggers, steps, approval gates, and guardrails
- Index skills: `python scripts/reindex_skills_simple.py`

### 5. Reindex Knowledge Base

```bash
python scripts/reindex_hotel_knowledge.py  # Adapt this script for your domain
```

---

## Documentation

- **[System Design Document](docs/system-design-document.md)** - Comprehensive architecture and design decisions
- **[Component Diagram](docs/component-diagram.md)** - Visual system overview
- **[Skills System Design](docs/skills-system-design.md)** - Skill schema and lifecycle
- **[RAG System](docs/rag-system-complete.md)** - Retrieval pipeline details
- **[Evaluation Framework](docs/evaluation-framework.md)** - Metrics and testing
- **[Installation Guide](docs/installation-guide.md)** - Detailed setup instructions

---

## Testing

```bash
# Run unit tests
pytest tests/unit/

# Run integration tests
pytest tests/integration/

# Run evaluation suite
python evals/run_system_evaluations.py
```

---

## Contributing

This is a proof-of-concept project. Contributions, suggestions, and adaptations are welcome!

---

## License

[Add your license here]

---

## Acknowledgments

Built as a capstone project to demonstrate:
- RAG-based knowledge grounding
- Agentic reasoning with ReAct pattern
- Human-in-the-loop AI systems
- Self-learning through skill compilation

**Current Status**: POC with hotel domain data  
**Intended Use**: Generic troubleshooting framework for any operational domain  
**Next Steps**: Executor Agent implementation, domain adapters, production integrations
