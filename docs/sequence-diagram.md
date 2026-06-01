# Customer Issue Resolution Copilot — Sequence Diagram

This diagram shows the main end-to-end flow for resolving a customer issue.

```mermaid
sequenceDiagram
    participant C as Customer Channel
    participant I as Issue Intake
    participant O as Orchestrator
    participant S as Skill Matcher
    participant R as RAG Service
    participant H as Human Reviewer
    participant E as Skill Executor
    participant A as ReAct Loop
    participant T as Tool Layer
    participant K as Skill Compiler
    participant G as Skills Registry

    C->>I: Submit issue from email or Slack
    I->>O: Normalize and forward issue
    O->>S: Check for matching skill
    S-->>O: Match result

    alt Existing skill found
        O->>R: Retrieve supporting context
        R-->>O: Ranked context
        O->>H: Present proposed resolution steps
        H-->>O: Approve execution
        O->>E: Execute approved skill
        E->>T: Call approved tools
        T-->>E: Return action result
        E-->>O: Execution outcome
        O-->>I: Final resolution steps or status
        I-->>C: Respond to support agent
    else No skill found
        O->>R: Retrieve relevant knowledge
        R-->>O: Ranked context
        O->>A: Start ReAct-style reasoning
        A->>T: Propose tool action
        T-->>A: Return observation
        A-->>O: Draft plan and next steps
        O->>H: Request approval for novel-task plan
        H-->>O: Approve or modify
        O->>T: Execute approved action
        T-->>O: Return result
        O->>K: Compile successful trace into draft skill
        K->>G: Save draft skill
        G-->>K: Confirm registration
        K-->>O: Draft skill created
        O-->>I: Resolution plus reusable draft skill
        I-->>C: Respond to support agent
    end
```

## Sequence Notes

- The system first tries to reuse an existing skill before invoking novel-task reasoning.
- Retrieval is used in both paths so that actions and responses remain grounded in company knowledge.
- Human approval is required before consequential actions.
- The novel-task path creates a reusable draft skill, which is the core learning mechanism of the POC.