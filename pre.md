Slide 1 — Cover（≈8s）

Good afternoon, professors. My name is **Mikuu** (Nie Wenhao). I’m a **4th-year CS student** in the  **SWJTU–University of Leeds joint program** , graduating in  **2026** . Today I’ll briefly share my work on  **reliable, auditable language intelligence** , especially when information involves  **multiple pieces of evidence and time constraints** .

---

### Slide 2 — Agenda（≈8s）

I’ll go through:  **background** , then two applied projects (**KLink** and  **JinkoRAG** ), then my  **final thesis on temporal reasoning** , and finally my  **future plan** .

---

### Slide 3 — Academic background（≈12s）

I study Computer Science with a strong engineering focus. My training is a mix of **system building** and  **model evaluation** , and I care a lot about turning model capability into something  **repeatable and testable** .

---

### Slide 4 — Achievements（≈12–15s）

I also have some research and competition experience, including a **patent under review** and several project outcomes. But for today, I’ll focus on what matters for this interview:  **how I build workflows, diagnose failures, and improve reliability** .

---

## KLink（Slides 5–6｜合计≈45–50s）

### Slide 5 — KLink background + related research（≈20–22s）

My first project is  **KLink** , a lesson-planning assistant for teachers. The core challenge is: teaching materials are scattered, and generated lesson plans can easily become  **off-topic or structurally inconsistent** .
So our direction is  **structured generation** : retrieve the right materials first, then generate step-by-step under a clear template. We also referenced work like **CRAG-Teach** and learning diagnosis ideas to make plans more targeted.

### Slide 6 — KLink contribution（≈25–28s）

My main contribution was the  **backend workflow orchestration** :

* I built a multi-step pipeline:  **Search → Generate → Constraints** .
* I used **hybrid retrieval + reranking** (vector + keyword signals) to improve relevance.
* And I did  **failure diagnosis** , like hallucination or inconsistent difficulty, and iterated the workflow based on feedback.
  This project convinced me that **making generation “workflow-based and constraint-based” is much more reliable** than free-form generation.

---

## JinkoRAG（Slides 7–9｜合计≈55–65s）

### Slide 7 — JinkoRAG background + related research（≈18–20s）

My second project is  **JinkoRAG** , a traceable QA system for enterprise documents, mainly for  **pre-sales contract rules** .
The key requirement is: answers must be **grounded** — they should  **point to evidence** , and if evidence is not enough, the system should **say “unknown”** rather than guess. This is closely related to **RAG attribution** and  **hallucination suppression** .

### Slide 8 — JinkoRAG contribution（≈20–22s）

I focused on three engineering pieces:

1. an **offline ingestion pipeline** with **incremental updates** (dedup, status, logs), so the knowledge base is maintainable;
2. a  **traceability constraint** , forcing the answer to cite sources and follow an “unknown if insufficient evidence” policy;
3. and a more **interpretable retrieval strategy** so we can debug why a document was retrieved.

### Slide 9 — JinkoRAG problem（≈20–25s）

And here’s the turning point: in real enterprise data, the same entity can have  **multiple valid periods** , or be **updated many times** with only  **small time changes** .
In that case, even if retrieval finds relevant chunks, the model often  **mixes versions** , confuses  **validity windows** , and produces an answer that sounds consistent but is actually  **temporally wrong** .
This pain point is exactly why I moved to my final thesis:  **temporal reasoning** .

---

## Final Thesis（Slides 10–12｜合计≈80–90s）

### Slide 10 — Thesis background + related research（≈25–28s）

My final thesis is about  **temporal reasoning for LLMs** . LLMs often fail on time constraints like **ordering, duration, time zones, and validity periods** — and in real systems, these are the reasons errors happen.
So I started from existing benchmarks and ideas, like **TRAM** and  **TempQuestions** , and also prompt methods like  **Chain-of-Thought** , plus constrained decoding evaluation, because in practice we need answers that are not only correct but also  **parsable and testable** .

### Slide 11 — What I did (progress + outcomes)（≈30–35s）

Instead of only “solving a dataset”, I’m building an **evaluation toolchain** — something you can run repeatedly across models and settings.
My current work includes:

* preparing task-wise data and sampling  **typical failure cases** ;
* making the system  **pluggable** , so different models and prompts can be compared fairly;
* and building reliable scoring by using  **answer extraction and normalization** , so the evaluation is stable rather than noisy.

### Slide 12 — Research plan (Gantt)（≈25–30s）

Going forward, my plan is a closed loop:  **evaluate → diagnose → improve** .
First, expand coverage and build **capability profiling** across sub-tasks and error types.
Second, introduce **structured intermediate representations** like  **timelines / event graphs** , so time constraints become explicit.
Third, run controlled experiments to verify whether this structure  **systematically reduces specific errors** , such as wrong timezone direction or validity conflicts.

---

### Slide 13 — Future plan（≈15–18s）

In parallel, I will strengthen my foundation and communication ability — including improving Japanese toward  **N2** , and continuing to improve my practical research execution, so I can contribute effectively in graduate research.

---

### Slide 14 — Thanks（≈8–10s）

That’s all for my presentation. Thank you very much, and I’m happy to take questions.

---
