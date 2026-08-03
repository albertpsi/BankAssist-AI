# Lab Evidence

The graded deliverable for this course is a **single completion document** covering all
seven labs. This directory holds the per-lab source material for it — written at the close
of each lab, while the friction is still fresh, then assembled at the end.

## Grading weights

| Component | Weight |
|---|---:|
| Lab completion | 50% |
| Technical implementation detail (screenshots, diagrams, code and configuration) | 35% |
| Learnings and reflection per lab | 15% |

The learnings section is 15% of the grade and is the part most likely to be reconstructed
badly from memory on the last day. Write it at each lab's exit.

## Files

| File | Lab | Status |
|---|---|---|
| `lab-01-ai-assisted-delivery.md` | AI-Assisted Software Delivery | ⬜ |
| `lab-02-basic-rag.md` | Basic RAG Pipeline | ⬜ |
| `lab-03-enterprise-rag.md` | Enterprise Multi-Stage RAG | ⬜ |
| `lab-04-multi-agent.md` | Multi-Agent Orchestration | ⬜ |
| `lab-05-guardrails.md` | Enterprise Guardrails Architecture | ⬜ |
| `lab-06-agentops-evaluation.md` | Automated Evaluation Pipeline | ⬜ |
| `lab-07-cost-optimization.md` | Cost Optimization Architecture | ⬜ |
| `submission.md` | Assembled completion document | ⬜ |

Screenshots go in `docs/labs/images/`, named `lab-NN-<what-it-shows>.png`.

## Per-lab template

Every section below is required by the lab brief. Sections marked ★ carry the most weight
relative to how easy they are to skip.

```markdown
# Lab N — <Title>

## Problem statement
What this lab is solving, in the context of BankAssist AI specifically.

## Objectives
What "done" means for this lab.

## Assumptions
What was taken as given, including anything about the synthetic dataset.

## Architecture approach
The design, with a Mermaid diagram. How it fits the layers built in previous labs.

## Key design decisions ★
Each decision, the alternatives, and why the alternative lost. Link the ADR if there is one.

## Implementation strategy
How it was actually built, in what order, and why that order.

## Validation approach
How correctness was established: tests written, what they assert, and the results.

## Screenshots and outputs ★
UI screenshots, trace excerpts, evaluation reports, benchmark tables.
Every claim in this document should have a corresponding artifact here.

## Code and configuration snippets ★
The 2–3 excerpts that best show how the capability actually works.
Prefer the interesting part over the boilerplate.

## Observations from implementation
What the system actually did, including behaviour that was surprising.

## Challenges encountered
What went wrong, and how it was diagnosed and resolved. Dead ends count — they are
evidence of real work.

## Trade-offs considered
What was given up, and what it bought.

## Lessons learned ★
The reflection section — 15% of the grade. What you would do differently, what transferred
to later labs, and what only became visible once the layer was running.

## Security, governance, scalability, operational and cost considerations
Where applicable to this lab. Be honest about what is *not* addressed.

## Recommendations for enterprise-scale adoption
What would have to change to run this for real. Draw on the scaling notes in
`docs/architecture/architecture.md §12` and `docs/architecture/technology-stack.md §5`.
```

## Capture checklist per lab

- [ ] Screenshots taken at the moment the capability first worked, not reconstructed later
- [ ] Trace excerpt showing the new layer's spans
- [ ] Test output pasted verbatim, including counts
- [ ] Any comparison table the lab's acceptance criteria require
- [ ] Challenges written down while the frustration is still specific
- [ ] Learnings written before starting the next lab
