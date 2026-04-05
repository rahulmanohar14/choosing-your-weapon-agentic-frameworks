# Chapter 11: Choosing Your Weapon

**LangGraph vs. AutoGen vs. CrewAI vs. PydanticAI in Production**

*Design of Agentic Systems with Case Studies*
INFO 7375: Prompt Engineering for Generative AI | Northeastern University

---

## Video

▶️ [Watch the 10-minute explainer on YouTube](https://youtu.be/rpvxvvI3Kkw)

The video follows the Explain → Show → Try structure required by the assignment rubric.

- **Explain (Scenes 1–5):** Architectural claim, four framework coordination models, topology diagrams
- **Show (Scenes 6–12):** Three-question procedure, decision matrix, LangGraph code walk, Human Decision Node on camera, CrewAI failure mechanistically, AutoGen variant, Loop Diagnostic
- **Try (Scenes 13–16):** Live notebook modification demonstrating the architectural gap

---

## What this repository contains

This repository is the companion implementation for Chapter 11 of *Design of Agentic Systems with Case Studies*. The chapter's core claim is: framework selection for multi-agent systems is an architectural decision driven by workflow dependency structure, and choosing based on popularity or feature count produces compounding integration debt when the framework's coordination model conflicts with the task's dependency graph.

This repository demonstrates that claim with runnable code. The same LLM (Llama 3.3 70B via Groq, temperature=0), the same task, the same system prompts, run in two different frameworks, produce different system behavior. The difference is not the model. It is the architecture.

---

## Repository structure

```
chapter-11-choosing-your-weapon/
│
├── ch11_choosing_your_weapon.md                         # Chapter (Markdown)
├── ch11_choosing_your_weapon.pdf                        # Chapter (PDF)
├── ch11_framework_selection_companion.ipynb             # Main demo notebook
├── Author's Note.docx                                     # Pedagogical report (3-page)
└── README.md                                            # This file
```

---

## The demo notebook

**File:** `ch11_framework_selection_companion.ipynb`

The notebook is divided into four parts. Each part maps to a section of the chapter.

### Part 1: Shared configuration (Chapter: The Mechanism)

Sets up the shared LLM used by both frameworks. This is the control variable. The same model runs both implementations so any behavioral difference is attributable to the coordination model, not the LLM.

```python
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
```

### Part 2: LangGraph implementation (Chapter: The Failure Case, LangGraph section)

Implements the three-node compliance pipeline as a directed state graph.

Key cells:
- State schema definition (PipelineState TypedDict)
- Agent function definitions (researcher, human_approval, writer)
- Graph construction with add_node, add_edge, add_conditional_edges
- Compilation with interrupt_before=["human_approval"]
- Execution: reject on first pass, approve on second pass

**What to observe:** The Writer node does not execute until approval_status equals "approved." The graph routes back to the Researcher on rejection. The Writer is topologically unreachable without a confirmed approval.

**Maps to chapter section:** "The Failure Case: The LangGraph Implementation"

### Part 3: CrewAI implementation (Chapter: The Failure Case, CrewAI section)

Implements the same pipeline as a sequential crew with role-based delegation.

Key cells:
- Agent definitions (researcher, reviewer, writer with human_input=True)
- Task definitions (research_task, review_task, writing_task)
- Crew assembly with Process.sequential
- Execution with hardcoded rejection

**What to observe:** The writing_task executes regardless of the reviewer's output. The FAILURE ANALYSIS section at the bottom of the execution cell prints the architectural explanation automatically. The notebook distinguishes between "Writer CANNOT execute" (LangGraph) and "Writer CHOSE not to execute" (CrewAI).

**Maps to chapter section:** "The Failure Case: The CrewAI Implementation"

### Part 4: Cascading failure with four agents (Chapter: Exercise 11.3)

Extends the CrewAI pipeline to four agents by adding a Regulatory Auditor. Demonstrates that CrewAI passes all prior task outputs to every downstream agent, meaning the Auditor receives rejected content it should never have access to. The LangGraph four-agent implementation passes only typed state fields to each node, enforcing input isolation architecturally.

**What to observe:** The Auditor's context report in the CrewAI run shows it received the original rejected research summary alongside the rejection message. In LangGraph the Auditor receives only the approved summary and the compliance memo.

**Maps to chapter section:** Exercise 11.3 "The Cascading Failure"

---

## The Try exercise notebook

**File:** `Chapter_11_Choosing_Your_Weapon_TRY_EXERCISE.ipynb`

This notebook contains the modification demonstrated in the video. It is a copy of the main notebook with the writer agent backstory changed to remove the BLOCKED instruction.

**The modification:**

Original writer backstory:
```python
backstory="""You are a technical writer specializing in regulatory communication. 
    IMPORTANT: You must ONLY draft memos based on APPROVED research. 
    If the research was rejected, you must NOT produce a memo. 
    Instead, output: 'BLOCKED: Cannot draft memo — research summary was rejected.'"""
```

Modified writer backstory:
```python
backstory="""You are a technical writer specializing in regulatory communication. 
    Draft a compliance memo based on the research summary provided."""
```

**What this demonstrates:** The original BLOCKED output was prompt-level compliance. The model chose to follow the instruction. Remove the instruction and the memo drafts anyway, because the architecture never prevented execution. In LangGraph, removing any instruction from the writer agent changes nothing. The graph topology prevents execution regardless of prompt content.

**This is the Try exercise from the video.** Run the crew.kickoff() cell in this notebook and observe the output. Then compare it to the LangGraph implementation running with the same rejection signal.

---

## Setup instructions

### Prerequisites

- Python 3.10 or higher
- Anaconda or conda environment manager
- A Groq API key (free tier at console.groq.com)

### Step 1: Clone the repository

```bash
git clone https://github.com/rahulmanohar14/choosing-your-weapon-agentic-frameworks.git
cd info7375-ch11-framework-selection
```

### Step 2: Create the conda environment

```bash
conda create -n chapter11 python=3.10
conda activate chapter11
```

### Step 3: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Set your Groq API key

Open the notebook and replace this line in Cell 1:

```python
os.environ["GROQ_API_KEY"] = "YOUR_GROQ_API_KEY_HERE"
```

With your actual Groq API key.

### Step 5: Run the notebook

```bash
jupyter lab
```

Open `ch11_framework_selection_companion.ipynb` and run all cells in order from top to bottom.

---

## Requirements

```
langchain-groq
langgraph
crewai
jupyter
jupyterlab
python-dotenv
```

Full pinned versions are in `requirements.txt`.

---

## The architectural claim in one sentence

The same LLM, the same temperature, the same system prompt, the same tool registry — attached to the wrong coordination model — produces a system that cannot be made correct without changing the architecture. The model cannot compensate for the architecture's structural limitations.

---

## Triggering the failure mode

To reproduce the CrewAI failure described in the chapter:

1. Open `Chapter_11_Choosing_Your_Weapon_TRY_EXERCISE.ipynb`
2. Run all setup cells (Cells 1 through 7)
3. Run the crew.kickoff() execution cell
4. Observe the FAILURE ANALYSIS output at the bottom
5. Then run the equivalent LangGraph cells and compare outputs

The contrast between the two outputs is the chapter's core argument made observable.

---

## Human Decision Node

This repository documents a correction made during research. The AI tool used to draft the chapter proposed that CrewAI has no native human-in-the-loop mechanism. This claim was rejected after verification against CrewAI's January 2026 changelog, which documents the Flows orchestration layer introducing conditional branching, typed state, and human interrupt points.

The correction is documented in:
- The chapter prose (Flows Evolution subsection)
- The Author's Note (HDN 3)
- The Mandatory Human Decision Node markdown cell in the notebook
- On camera in the video (Scene 9) — [watch here](https://youtu.be/rpvxvvI3Kkw)

Source: CrewAI Changelog, January 8, 2026. https://docs.crewai.com/en/changelog

---

## Chapter map

| Chapter section | Notebook location |
|---|---|
| The Scenario | Notebook introduction markdown |
| The Mechanism: LangGraph | Part 1 and Part 2 setup cells |
| The Mechanism: CrewAI | Part 3 setup cells |
| The Failure Case: LangGraph | Part 2 execution cells |
| The Failure Case: CrewAI | Part 3 execution cells |
| The AutoGen Variant | Described in chapter, not demonstrated in notebook |
| The Integration Debt Calculation | Part 3 FAILURE ANALYSIS output |
| The Loop Diagnostic | Part 3 and Part 4 comparison |
| Exercise 11.3: The Cascading Failure | Part 4 cells |
| Exercise 11.4: The Selection Audit | Try exercise notebook |

---

## Author

Rahul Manohar Durshinapally
Northeastern University
