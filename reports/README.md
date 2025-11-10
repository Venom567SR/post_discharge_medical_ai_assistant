# Reports Overview

This directory contains documentation related to the system design and reasoning behind the Post-Discharge Medical AI Assistant. The project is built around a multi-agent medical assistance workflow featuring:

- **LangGraph** for deterministic agent orchestration
- **RAG** over a nephrology reference PDF for evidence-backed guidance
- **Gemini 2.5 Flash** as the primary LLM and **Groq LLaMA 3.1 8B Instruct** as fallback
- **Qdrant** as the primary vector store, with **Chroma** available as a lightweight local fallback

---

## Files in This Directory

| File | Format | Purpose |
|------|--------|---------|
| **Architecture_Brief.docx** | `.docx` | High-level system workflow and component overview |
| **DESIGN_RATIONALE.md** | `.md` | Rationale for key technology and model choices |

> A demo video is provided separately during submission (not included in this repository to avoid large binary storage).

---

## Suggested Review Order

1. **DESIGN_RATIONALE.md**  
   Outlines the trade-offs and decision-making behind LLMs, vector DBs, and agent frameworks.

2. **Architecture_Brief.docx**  
   Explains how system components interact and how control/handoff flows between agents are structured.

---

## Key Design Principles

- **Grounded Medical Reasoning**: Clinical information is retrieved from reference material rather than generated freely.
- **Explicit Role Separation**: The Receptionist Agent handles identification and intake; the Clinical Agent handles medical reasoning.
- **Deterministic Agent Control**: LangGraph ensures predictable and auditable state transitions.
- **Fail-Operational Behavior**: If model or web search APIs are unavailable, the assistant continues functioning using safe stub or fallback paths.

---

## Relevant References

- **LLaMA 3 Model Family Overview**  
  https://arxiv.org/abs/2407.21783  
  Describes the 8B Instruct model used via Groq for fast, low-latency inference.

- **Gemini 2.5 Flash Technical Report**  
  https://storage.googleapis.com/deepmind-media/gemini/gemini_2_5_1120.pdf  
  Highlights structured-output capabilities and reasoning performance relevant to medical guidance.

- **Survey on Vector Databases**  
  https://arxiv.org/abs/2310.11703  
  Discusses design considerations for vector retrieval systems such as Qdrant.

---

This documentation provides context for the architecture decisions that shape system behavior, safety assumptions, and reasoning transparency.