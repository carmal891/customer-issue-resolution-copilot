# Customer Issue Resolution Copilot — Component Diagram

This diagram gives a TA-friendly high-level view of the major system components and how they interact.

```mermaid
flowchart TD
    U[Customer via email or Slack] --> I[Issue Intake Layer]
    I --> O[Orchestrator]

    O --> SM[Skill Matcher]
    O --> RAG[RAG Retrieval Service]

    RAG --> KB[Knowledge Base]
    KB --> DOCS[Policies and Runbooks]
    KB --> SLACK[Slack Threads]
    KB --> JIRA[Jira Tickets]
    KB --> VDB[Vector Index]

    SM --> SR[Skills Registry]
    SR --> YAML[Skill YAML Files]
    SR --> META[Skill Metadata and Embeddings]

    O --> DECIDE{Skill found}

    DECIDE -->|Yes| SE[Skill Executor]
    DECIDE -->|No| REACT[ReAct-style Loop]
    REACT --> TPAO[TPAO Implementation]

    SE --> TOOLS[Approved Tool Layer]
    REACT --> TOOLS

    TOOLS --> DB[Operational Actions]
    TOOLS --> MAIL[Notifications]
    TOOLS --> ESC[Escalation Artifacts]

    SE --> HITL[Human Approval Interface]
    REACT --> HITL
    HITL --> TOOLS

    REACT --> COMP[Skill Compiler]
    COMP --> DRAFT[Draft Skill]
    DRAFT --> SR

    O --> OUT[Resolution Steps or Escalation Output]
```

## Component Notes

- **Issue Intake Layer** receives mock customer issues from email or Slack.
- **Orchestrator** decides whether the issue can be handled by an existing skill or needs novel-task reasoning.
- **RAG Retrieval Service** grounds the system using company knowledge.
- **Skill Matcher** checks whether a reusable skill already exists.
- **ReAct-style Loop** is the recognizable reasoning pattern used for novel tasks.
- **TPAO Implementation** is the project-specific realization of the ReAct loop.
- **Human Approval Interface** ensures risky actions are reviewed before execution.
- **Skill Compiler** converts successful novel traces into reusable draft skills.