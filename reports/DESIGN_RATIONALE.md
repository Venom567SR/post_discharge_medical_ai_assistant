# Design Rationale & Technology Choices

This document explains the key technical choices in the Post-Discharge Medical AI Assistant and the reasoning behind them. The system prioritizes clarity, reliability, and reproducibility over experimental complexity, and emphasizes components that are practical and aligned with real-world deployment needs.

---

## 1. Language Models (LLMs)

### Primary Model: Gemini 2.5 Flash

**Reasons**
- Strong reasoning ability and structured output support.
- Performs well when grounded through RAG.
- Free-tier availability allows the system to run without paid API usage.
- Responsive for conversational workflows.

**Trade-offs**
- Slightly weaker than flagship GPT models in nuanced clinical reasoning.
- Requires grounding with citations to avoid hallucination.

### Fallback Model: Groq LLaMA 3.1 8B Instant

**Reasons**
- Extremely low latency (<100ms typical).
- High coherence for general responses.
- Free via Groq API key.
- Prevents model failures from interrupting conversation flow.

**Fallback Strategy**
- Try Gemini first.
- If unavailable, use Groq.
- Ensures the assistant always responds.

### GPT‑5‑mini Consideration

- Initially preferred for reliability and reasoning stability.
- Not used due to paid API requirements for continuous operation.
- Gemini chosen to support reproducibility for evaluators.

---

## 2. Vector Database

### Qdrant (Primary)

**Reasons**
- Production-ready and stable.
- Efficient CPU performance.
- Supports metadata filtering and persistent storage.
- Widely adopted in RAG pipelines.

### Chroma (Fallback)

**Reasons**
- Lightweight and easy to run locally.
- Useful for rapid testing or demonstration environments.
- Lower operational friction.

**Rationale**
Qdrant is chosen for scalability; Chroma supports ease-of-use for reviewers.

---

## 3. Agent Framework

### LangGraph (Primary)

**Reasons**
- Deterministic state transitions.
- Supports transparent agent routing.
- Easier to debug via graph execution view.
- Suitable for controlled medical-support workflows.

### CrewAI (Not chosen for main pipeline)

**Reasons**
- Useful for prototype exploration but less deterministic.
- Harder to enforce structured conversational turns.
- Ensuring reproducibility is more difficult.

---

## 4. Patient Data Format: JSON Records

**Reasons**
- Easy to manage and version control.
- Works well with FastAPI and Pydantic validation.
- Sufficient for demonstration use without requiring a full database layer.

---

## 5. Backend & Frontend

### FastAPI (Backend)
- Typed request/response validation.
- Auto-generated documentation at `/docs`.
- Production-ready.

### Streamlit (Frontend)
- Quick development of conversational UI.
- Avoids additional JS or frontend frameworks.

---

## 6. System Goals

The system prioritizes:

| Goal | Description |
|------|-------------|
| Clarity | Predictable, explainable agent behavior |
| Safety | RAG grounding + citation transparency |
| Resilience | Stable fallback behavior to avoid failure modes |
| Accessibility | Works even without paid API keys |

---

## Summary

| Component | Choice | Key Advantage |
|---------|--------|----------------|
| Primary LLM | Gemini 2.5 Flash | Capable + accessible to run |
| Fallback LLM | Groq LLaMA 3.1 8B | Extremely low latency fallback |
| Vector DB | Qdrant | Reliable and scalable |
| Agent Framework | LangGraph | Controlled multi-agent orchestration |
| UI | Streamlit | Simple interactive interface |
| Backend | FastAPI | Structured and deployable |

This design reflects practical engineering trade-offs and emphasizes safe, interpretable, and reproducible behavior.
