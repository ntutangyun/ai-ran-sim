# AI RAN Simulation Evaluation Framework

This directory contains a comprehensive, autonomous evaluation framework for multi-agent AI systems in telecom simulation, with full support for HATT-E academic evaluation.

---

## Overview: What Does This Evaluation Do?

- **Autonomous Agent Testing:**
  - `run_evaluation.py` is a fully autonomous agent testing framework.
  - It uses AI to drive realistic, multi-turn conversations with your agent(s), simulating real-world network engineer queries.
  - The framework evaluates agent responses using both tool outputs and AI-based self-evaluation, producing detailed logs, scores, and reasoning for every question.
  - Results are saved in timestamped folders for easy review and comparison.

- **Automated Visualization:**
  - `visualization_engine.py` generates all performance charts and dashboards automatically when run (no arguments needed).
  - It covers overall performance, difficulty/category breakdowns, tool usage, error handling, metric radar, and more.
  - The radar chart is now called `metric_radar` for clarity and generality.

# HATT-E Evaluation Framework: README

## What is HATT-E?
HATT-E (Hierarchical Agent Task and Tool Evaluation) is a comprehensive, multi-layered evaluation framework for multi-agent AI systems, designed for complex agentic environments. It provides both quantitative and qualitative metrics to assess the performance, reliability, and reasoning of AI agents at different levels of orchestration and specialization.

---

## Why HATT-E?
Modern AI systems often use multiple agents (orchestrators, specialists, tool-users) to solve complex tasks. Evaluating such systems requires more than just task success rates—it requires understanding how well the system decomposes tasks, delegates, executes, and collaborates. HATT-E provides a structured, academic approach to this evaluation.

---

## HATT-E Layers Overview

### **Layer 1: Orchestration/Decomposition**
- **What:** Evaluates the top-level agent (orchestrator/planner) on how well it understands user intent, decomposes tasks, and delegates to specialists.
- **Why:** The quality of decomposition and delegation directly impacts the system's ability to solve complex, multi-step problems.
- **Metrics Implemented:**
  - **Decomposition Quality Score (DQS):** Logical coherence, completeness, and efficiency of the plan.
  - **Delegation Accuracy:** Did the orchestrator assign sub-tasks to the correct specialists/tools?
  - **Failure Mode Tagging:** Identifies common orchestration errors (e.g., missing steps, wrong delegation).
- **What we've done:**
  - Automated extraction and scoring of decomposition plans.
  - Aggregated all Layer 1 results and visualizations in a consistent output structure.
  - Robust error handling and clear reporting.

### **Layer 2: Specialist/Tool Proficiency**
- **What:** Evaluates the specialist agents and tool-using agents on how well they execute the plan, use tools, and avoid errors like hallucination.
- **Why:** Even a perfect plan fails if the specialists/tools do not execute it correctly. Layer 2 measures the system's ability to follow through on the orchestrator's intent.
- **Key Metrics Implemented:**
  - **Plan.EM (Plan Execution Match):**
    - Compares the orchestrator's intended tool usage (from the plan) to the actual tool calls made during execution.
    - Uses both heuristic (Jaccard similarity) and LLM-based scoring for robust evaluation.
    - Provides detailed reasoning for each score.
  - **Act.EM (Action Execution Match):**
    - Compares the planned tools (from `expected_tools`) to the tools actually used (`tools_used`) for each question.
    - Uses Jaccard similarity and provides clear reasoning for each score.
    - Answers: "Did the system actually perform the actions it was supposed to?"
- **What we've done:**
  - Automated extraction and scoring for both Plan.EM and Act.EM.
  - Always use LLM for Plan.EM for interpretability and robustness.
  - Output per-question and aggregate results, with clear reasoning and summary statistics.

---

## Why is Layer 2 Important?
- **Bridges the gap between planning and execution.**
- **Identifies breakdowns:** If Layer 1 is perfect but Layer 2 is poor, the problem is in execution, not planning.
- **Academic rigor:** Provides fine-grained, explainable metrics for research and engineering improvement.

---

## How to Use This Evaluation
1. **Run the evaluation scripts:**
   - Layer 1 and Layer 2 can be run independently via CLI.
   - Results are saved in structured JSON files for further analysis or visualization.
2. **Interpret the results:**
   - Use the reasoning fields to understand why scores were given.
   - Aggregate metrics help identify systemic strengths and weaknesses.

---

## Summary Table
| Layer   | Metric         | What it Measures                                 | How it's Scored         |
|---------|---------------|--------------------------------------------------|-------------------------|
| Layer 1 | DQS           | Plan quality (coherence, completeness, efficiency)| LLM/heuristic           |
| Layer 1 | Delegation    | Correctness of sub-task assignment               | Heuristic/LLM           |
| Layer 2 | Plan.EM       | Match between planned and executed tool usage    | Jaccard + LLM           |
| Layer 2 | Act.EM        | Match between planned and actual actions         | Jaccard + Reasoning     |
| Layer 2 | TSR           | Task success rate (binary success/failure)       | Heuristic               |
| Layer 2 | Hallucination | Rate of incorrect information generation         | LLM-based detection     |
| Layer 3 | TSR           | Task success rate with LLM judgment              | LLM-as-a-judge          |
| Layer 3 | Response Quality| Quality of agent responses (1-5 scale)          | LLM-as-a-judge          |
| Layer 3 | Consistency   | Consistency across similar questions              | LLM-as-a-judge          |
| Layer 3 | System Cost   | API calls, tokens, processing time efficiency    | Calculated metrics       |
| Layer 3 | Latency       | Response time and processing efficiency           | Calculated metrics       |
| Layer 3 | Turn Count    | Conversation efficiency and complexity            | Calculated metrics       |

---

## For More Information
- See the code in `hatt_e_metrics.py` for implementation details.
- All results and reasoning are saved in the `hatt_e/layer1/` and `hatt_e/layer2/` output folders.
- For academic citation or further reading, see the research papers related to HATT-E framework or contact the project maintainers. 
---

## Setup Instructions

1. **Create a virtual environment:**
   ```bash
   python3 -m venv backend/.venv
   source backend/.venv/bin/activate
   ```
2. **Install dependencies:**
```bash
   pip install -r backend/requirements.txt
```
3. **Set your OpenAI API key:**
```bash
export OPENAI_API_KEY="your-api-key-here"
```

---

## How to Use

### 1. Run the Autonomous Evaluation
```bash
python backend/evaluation/run_evaluation.py
```
- This will:
  - Initialize the simulation and agents
  - Run dynamic, AI-driven conversations for all questions in `conversation_data.json`
  - Evaluate every response, log all tool usage, and save detailed results in a new `evaluation_results_YYYYMMDD_HHMMSS/` folder

### 2. Generate All Visualizations
```bash
python backend/evaluation/visualization_engine.py evaluation_results_YYYYMMDD_HHMMSS
```
- This will:
  - Automatically generate all charts and dashboards (no arguments needed)
  - Save plots (including `metric_radar.png`) in the results folder

### 3. (Optional) Run HATT-E Metrics
```bash
python backend/evaluation/hatt_e_metrics.py evaluation_results_YYYYMMDD_HHMMSS
```
- Computes complete HATT-E evaluation (Layer 1 + Layer 2 + Layer 3) with comprehensive visualizations.
- Automatically generates all metrics, visualizations, and aggregations in one command.

---

## HATT-E Evaluation: Layers Explained

- **Layer 1: Orchestration/Decomposition**
  - Evaluates how well the orchestrator agent breaks down user requests, delegates to specialists/tools, and handles failure modes.
  - Metrics: Decomposition Quality Score (DQS), Delegation Accuracy, Failure Mode Tagging
  - Outputs: Per-question and aggregate stats, DQS histograms, delegation accuracy plots, failure mode barplots

- **Layer 2: Specialist/Tool**
  - Evaluates how well specialist agents/tools execute the orchestrator’s plan and produce correct results.
  - Metrics: Plan Execution Match (Plan.EM), Action Execution Match (Act.EM), Tool Success Rate (TSR), Hallucination Rate
  - Outputs: (Planned) Per-question and aggregate stats, visualizations

- **Layer 3: System/Collaboration**
  - Evaluates overall system performance, collaboration, user outcomes, and task success.
  - Metrics: Task Success Rate (LLM-judged), Response Quality (1-5 scale), Consistency, System Cost, Latency, Turn Count
  - Outputs: Per-question and aggregate stats, comprehensive visualizations, correlation analysis, performance dashboard

---

## Output Structure

- `evaluation_results_YYYYMMDD_HHMMSS/`
  - `conversation_q_XXX.log` — Full logs for each question
  - `dynamic_conversation_evaluation_*.json` — Detailed evaluation results
  - `evaluation_summary_*.json` — High-level summary
  - Plots: `overall_performance.png`, `metric_radar.png`, etc.
  - HATT-E metrics: `hatt_e/layer1/`, `hatt_e/layer2/`, `hatt_e/layer3/` folders with comprehensive results
  - Layer 3 visualizations: `hatt_e/visualization/` with correlation plots, performance dashboards

---

## Customization & Extension
- **Edit questions:** Modify `conversation_data.json`.
- **Add metrics:** Extend `hatt_e_metrics.py` or `visualization_engine.py`.
- **Layer 3:** Fully implemented with comprehensive metrics and visualizations.
- **Academic use:** All metrics provide detailed reasoning and are suitable for research publication.

---

## Questions?
- See code comments and docstrings for details.
- For academic use, cite the HATT-E framework and this repository. 