import os
import json
from typing import List, Dict, Any, Optional
import asyncio
import re
import ast
from collections import Counter, defaultdict
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Import openai_agents_sdk Agent and Runner
from agents import Agent, Runner
from agents import set_default_openai_key, set_tracing_disabled
set_default_openai_key(os.environ.get("OPENAI_API_KEY", ""))
set_tracing_disabled(True)

class HATTEvaluator:
    """
    Computes HATT-E Layer 1 metrics from evaluation results and logs.
    Designed for use with /evaluation_results_*/ structure.
    """
    def __init__(self, results_dir: str):
        self.results_dir = results_dir
        self.output_dir = os.path.join(results_dir, "hatt_e", "layer1")
        self._ensure_output_dir()
        self.conversation_data_path = self._find_conversation_data()
        self.questions = self._load_questions()
        self.dynamic_eval = self._load_dynamic_eval()
        self.log_files = self._find_log_files()

    def _ensure_output_dir(self):
        os.makedirs(self.output_dir, exist_ok=True)

    def _find_conversation_data(self) -> str:
        # Look for conversation_data.json in results_dir, then parent
        local = os.path.join(self.results_dir, "conversation_data.json")
        parent = os.path.join(os.path.dirname(self.results_dir), "conversation_data.json")
        if os.path.exists(local):
            return local
        elif os.path.exists(parent):
            return parent
        else:
            raise FileNotFoundError("conversation_data.json not found in results_dir or its parent.")

    def _load_questions(self) -> Dict[str, Any]:
        with open(self.conversation_data_path, 'r') as f:
            data = json.load(f)
        return {q['id']: q for q in data['questions']}

    def _load_dynamic_eval(self) -> Dict[str, Any]:
        for fname in os.listdir(self.results_dir):
            if fname.startswith('dynamic_conversation_evaluation_') and fname.endswith('.json'):
                with open(os.path.join(self.results_dir, fname), 'r') as f:
                    return json.load(f)
        return {}

    def _find_log_files(self) -> Dict[str, str]:
        files = {}
        indiv_log_dir = os.path.join(self.results_dir, 'individual_conversation_logs')
        if not os.path.exists(indiv_log_dir):
            return files
        for fname in os.listdir(indiv_log_dir):
            if fname.startswith('conversation_q_') and fname.endswith('.log'):
                qid = 'q_' + fname.split('_q_')[1].split('.')[0].zfill(3)
                files[qid] = os.path.join(indiv_log_dir, fname)
        return files

    def _load_log(self, question_id: str) -> Optional[str]:
        log_path = self.log_files.get(question_id)
        if not log_path or not os.path.exists(log_path):
            return None
        with open(log_path, 'r') as f:
            return f.read()

    def extract_user_request_and_plan(self, qid: str) -> Optional[Dict[str, str]]:
        dyn_eval = self._find_dynamic_eval_for_qid(qid)
        if not dyn_eval:
            return None
        user_request = dyn_eval.get('static_question', '')
        plan = ''
        conversation = dyn_eval.get('dynamic_conversation', [])
        for i, turn in enumerate(conversation):
            if turn['role'] == 'user':
                if i + 1 < len(conversation) and conversation[i+1]['role'] == 'assistant':
                    plan = conversation[i+1]['content']
                    break
        if not plan:
            plan = dyn_eval.get('agent_response', '')
        return {'user_request': user_request, 'decomposition_plan': plan}

    def generate_dqs_llm_prompt(self, user_request: str, decomposition_plan: str) -> str:
        prompt = f"""
You are an expert AI system evaluator. Your task is to assess the quality of a task decomposition plan created by an orchestrator agent based on a user request.

**User Request:**
{user_request}

**Orchestrator's Decomposition Plan:**
{decomposition_plan}

**Evaluation Criteria & Rubric:**
1. Logical Coherence (1-5): Is the breakdown logical and sensible?
2. Completeness (1-5): Does it fully address all aspects of the user's request?
3. Efficiency (1-5): Is the breakdown efficient, or does it contain redundant steps?

First, provide a step-by-step analysis of the decomposition plan against each criterion, explaining your reasoning for the scores.
Second, provide a numerical score for each criterion.
Finally, output the scores in a single JSON object: {{"coherence": <score>, "completeness": <score>, "efficiency": <score>}}.
"""
        return prompt

    async def evaluate_dqs_llm_for_all(self, output_path: str):
        """
        For each question, call the DQS LLM judge and store the scores.
        """
        # Create the DQS judge agent
        dqs_judge_agent = Agent(
            name="DQS Judge Agent",
            instructions="""You are an expert AI system evaluator. You will be given a user request and an orchestrator's decomposition plan. Your job is to evaluate the plan using a rubric and output a JSON with scores for coherence, completeness, and efficiency.""",
            model=os.environ.get("OPENAI_NON_REASONING_MODEL_NAME", "gpt-3.5-turbo")
        )
        results = []
        for qid in self.questions:
            extracted = self.extract_user_request_and_plan(qid)
            if not extracted:
                print(f"[WARN] Could not extract for {qid}")
                continue
            prompt = self.generate_dqs_llm_prompt(extracted['user_request'], extracted['decomposition_plan'])
            # Call the agent
            print(f"[DQS LLM] Scoring {qid}...")
            runner = Runner.run_streamed(dqs_judge_agent, prompt)
            response_text = ""
            async for event in runner.stream_events():
                if event.type == "raw_response_event":
                    delta = getattr(event.data, "delta", None)
                    if isinstance(delta, str):
                        response_text += delta
                    elif isinstance(event.data, str):
                        response_text += event.data
                elif event.type == "message_output_item":
                    if isinstance(event.data, str):
                        response_text += event.data
                elif event.type == "run_completed":
                    break
            # Try to extract the JSON from the response
            json_match = re.search(r'\{\s*"coherence".*?\}', response_text, re.DOTALL)
            dqs_scores = {"coherence": None, "completeness": None, "efficiency": None}
            if json_match:
                try:
                    dqs_scores = ast.literal_eval(json_match.group(0))
                except Exception as e:
                    print(f"[WARN] Could not parse DQS JSON for {qid}: {e}")
            else:
                print(f"[WARN] No DQS JSON found in LLM response for {qid}")
            result = {
                "question_id": qid,
                "dqs_llm_response": response_text,
                "dqs_scores": dqs_scores
            }
            results.append(result)
        # Save results
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"[DQS LLM] All results written to {output_path}")

    def print_dqs_prompts_for_all(self):
        for qid in self.questions:
            extracted = self.extract_user_request_and_plan(qid)
            if not extracted:
                print(f"[WARN] Could not extract for {qid}")
                continue
            prompt = self.generate_dqs_llm_prompt(extracted['user_request'], extracted['decomposition_plan'])
            print(f"\n--- DQS LLM Prompt for {qid} ---\n{prompt}\n{'-'*60}")

    def evaluate_layer1(self) -> List[Dict[str, Any]]:
        results = []
        for qid, q in self.questions.items():
            log = self._load_log(qid)
            if not log:
                continue
            dyn_eval = self._find_dynamic_eval_for_qid(qid)
            result = {"question_id": qid}
            # 1. Decomposition Quality Score (DQS)
            result["dqs"] = self._compute_dqs(log, q, dyn_eval)
            # 2. Delegation Accuracy (DA)
            result["delegation_accuracy"] = self._compute_delegation_accuracy(log, q, dyn_eval)
            # 3. Handoff Information Fidelity (HIF)
            result["handoff_info_fidelity"] = self._compute_handoff_info_fidelity(log, q, dyn_eval)
            results.append(result)
        return results

    def _find_dynamic_eval_for_qid(self, qid: str) -> Optional[Dict[str, Any]]:
        if isinstance(self.dynamic_eval, dict) and "detailed_results" in self.dynamic_eval:
            for r in self.dynamic_eval["detailed_results"]:
                if r.get("question_id") == qid:
                    return r
        elif isinstance(self.dynamic_eval, list):
            for r in self.dynamic_eval:
                if r.get("question_id") == qid:
                    return r
        return None

    def _compute_dqs(self, log: str, question: Dict[str, Any], dyn_eval: Optional[Dict[str, Any]]) -> float:
        if "plan" in log.lower() or "Based on the current configuration" in log:
            return 1.0
        return 0.5

    def _compute_delegation_accuracy(self, log: str, question: Dict[str, Any], dyn_eval: Optional[Dict[str, Any]]) -> float:
        expected_tools = set(question.get("expected_tools", []))
        used_tools = set()
        if dyn_eval and "tools_used" in dyn_eval:
            used_tools = set(dyn_eval["tools_used"])
        else:
            tools_line = next((line for line in log.splitlines() if line.startswith("TOOLS USED:")), None)
            if tools_line:
                try:
                    used_tools = set(eval(tools_line.split(":", 1)[1].strip()))
                except Exception:
                    used_tools = set()
        if not expected_tools:
            return 1.0 if not used_tools else 0.0
        correct = len(expected_tools & used_tools)
        total = len(expected_tools)
        return correct / total if total > 0 else 1.0

    def _compute_handoff_info_fidelity(self, log: str, question: Dict[str, Any], dyn_eval: Optional[Dict[str, Any]]) -> float:
        required_keywords = question.get("evaluation_criteria", {}).get("expected_tool_output_contains", [])
        present = sum(1 for kw in required_keywords if kw.lower() in log.lower())
        if not required_keywords:
            return 1.0
        return present / len(required_keywords)

    def compute_delegation_accuracy_report(self, output_path: str):
        """
        For each question, compare tools_used (from dynamic eval) to expected_tools (from conversation_data),
        and score as the fraction of expected tools that were actually used.
        Output a JSON report.
        """
        results = []
        for qid, q in self.questions.items():
            dyn_eval = self._find_dynamic_eval_for_qid(qid)
            expected_tools = set(q.get("expected_tools", []))
            used_tools = set()
            if dyn_eval and "tools_used" in dyn_eval:
                used_tools = set(dyn_eval["tools_used"])
            # Score: fraction of expected tools that were actually used
            if not expected_tools:
                score = 1.0 if not used_tools else 0.0
            else:
                correct = len(expected_tools & used_tools)
                total = len(expected_tools)
                score = correct / total if total > 0 else 1.0
            result = {
                "question_id": qid,
                "expected_tools": list(expected_tools),
                "tools_used": list(used_tools),
                "delegation_accuracy": score
            }
            results.append(result)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"[Delegation Accuracy] Results written to {output_path}")

    def tag_failure_modes(self, dqs_llm_path: str, output_path: str):
        """
        For each question, tag Layer 1 failure modes based on the DQS LLM response.
        Output a JSON report with tags for each question.
        """
        with open(dqs_llm_path, 'r') as f:
            dqs_data = json.load(f)
        results = []
        for entry in dqs_data:
            qid = entry["question_id"]
            response = entry.get("dqs_llm_response", "").lower()
            tags = set()
            # Flawed decomposition
            if re.search(r"illogical|not logical|lacks logical|logical coherence: 1|logical coherence: 2", response):
                tags.add("flawed_decomposition")
            # Incomplete
            if re.search(r"incomplete|does not fully address|missing|not fully|partially addresses|completeness: 1|completeness: 2", response):
                tags.add("incomplete")
            # Redundant/inefficient
            if re.search(r"redundant|unnecessary|inefficient|efficiency: 1|efficiency: 2", response):
                tags.add("redundant_steps")
            # Incorrect tool (if mentioned)
            if re.search(r"incorrect tool|wrong tool|did not use the expected tool|mismatch in tool usage", response):
                tags.add("incorrect_tool")
            # If all scores are 4 or 5 and no negative keywords, tag as correct
            scores = entry.get("dqs_scores", {})
            if (all(scores.get(k, 0) >= 4 for k in ["coherence", "completeness", "efficiency"]) and not tags):
                tags.add("correct")
            # If no tags, but scores are low, tag as 'other_issue'
            if not tags and any(scores.get(k, 0) < 4 for k in ["coherence", "completeness", "efficiency"]):
                tags.add("other_issue")
            results.append({
                "question_id": qid,
                "failure_mode_tags": list(tags),
                "dqs_scores": entry.get("dqs_scores", {}),
                "dqs_llm_response": entry.get("dqs_llm_response", "")
            })
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"[Failure Mode Tagging] Results written to {output_path}")

    def aggregate_layer1_results(self, dqs_llm_path: str, da_path: str, failure_modes_path: str, output_path: str):
        """
        Aggregate all Layer 1 metrics and stats into a single JSON output.
        """
        with open(dqs_llm_path, 'r') as f:
            dqs_data = json.load(f)
        with open(da_path, 'r') as f:
            da_data = json.load(f)
        with open(failure_modes_path, 'r') as f:
            fm_data = json.load(f)

        # Index by question_id
        dqs_by_qid = {x['question_id']: x for x in dqs_data}
        da_by_qid = {x['question_id']: x for x in da_data}
        fm_by_qid = {x['question_id']: x for x in fm_data}

        all_qids = sorted(set(dqs_by_qid) | set(da_by_qid) | set(fm_by_qid))
        per_question = []
        for qid in all_qids:
            entry = {
                'question_id': qid,
                'dqs_scores': dqs_by_qid.get(qid, {}).get('dqs_scores'),
                'delegation_accuracy': da_by_qid.get(qid, {}).get('delegation_accuracy'),
                'failure_mode_tags': fm_by_qid.get(qid, {}).get('failure_mode_tags'),
            }
            per_question.append(entry)

        # Aggregate stats
        dqs_scores = defaultdict(list)
        da_scores = []
        fm_counter = Counter()
        for q in per_question:
            if q['dqs_scores']:
                for k, v in q['dqs_scores'].items():
                    dqs_scores[k].append(v)
            if q['delegation_accuracy'] is not None:
                da_scores.append(q['delegation_accuracy'])
            if q['failure_mode_tags']:
                for tag in q['failure_mode_tags']:
                    fm_counter[tag] += 1

        stats = {
            'dqs_averages': {k: sum(v)/len(v) if v else None for k, v in dqs_scores.items()},
            'delegation_accuracy_avg': sum(da_scores)/len(da_scores) if da_scores else None,
            'failure_mode_counts': dict(fm_counter),
        }

        output = {
            'per_question': per_question,
            'aggregate_stats': stats
        }
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)

    def visualize_layer1_metrics(self):
        """
        Generate and save visualizations for Layer 1 metrics in /visualization/ with _hatte_l1.png suffix.
        """
        # Paths
        vis_dir = os.path.join(os.path.dirname(self.output_dir), "visualization")
        os.makedirs(vis_dir, exist_ok=True)
        agg_path = os.path.join(self.output_dir, "hatt_e_layer1_aggregated.json")
        with open(agg_path, "r") as f:
            data = json.load(f)
        per_question = data["per_question"]
        stats = data["aggregate_stats"]

        # DQS histograms
        for metric in ["coherence", "completeness", "efficiency"]:
            values = [q["dqs_scores"][metric] for q in per_question if q["dqs_scores"] and q["dqs_scores"].get(metric) is not None]
            plt.figure(figsize=(6,4))
            sns.histplot(values, bins=range(1,7), kde=False, discrete=True)
            plt.title(f"DQS {metric.capitalize()} Histogram")
            plt.xlabel(f"{metric.capitalize()} Score")
            plt.ylabel("Count")
            plt.tight_layout()
            plt.savefig(os.path.join(vis_dir, f"dqs_{metric}_histogram_hatte_l1.png"))
            plt.close()

        # Delegation Accuracy histogram
        da_values = [q["delegation_accuracy"] for q in per_question if q["delegation_accuracy"] is not None]
        plt.figure(figsize=(6,4))
        sns.histplot(da_values, bins=11, kde=False)
        plt.title("Delegation Accuracy Histogram")
        plt.xlabel("Delegation Accuracy")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, "delegation_accuracy_histogram_hatte_l1.png"))
        plt.close()

        # Failure Mode barplot
        fm_counts = stats["failure_mode_counts"]
        plt.figure(figsize=(7,4))
        sns.barplot(x=list(fm_counts.keys()), y=list(fm_counts.values()))
        plt.title("Failure Mode Counts")
        plt.xlabel("Failure Mode Tag")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, "failure_mode_barplot_hatte_l1.png"))
        plt.close()

        print(f"[HATT-E] Layer 1 visualizations saved to {vis_dir}")

    def _extract_tools_from_plan(self, plan: str) -> List[str]:
        """
        Extract tool/specialist names from the plan using regex/heuristics.
        If ambiguous, fallback to LLM (not implemented here, placeholder for future).
        """
        # Simple heuristic: look for tool names in curly braces or as keywords
        # Example: {"knowledge_query_key": ...} or 'get_knowledge', 'query', etc.
        tool_pattern = re.compile(r'"([a-zA-Z0-9_]+_query|get_[a-zA-Z0-9_]+|[a-zA-Z0-9_]+_tool)"')
        found = tool_pattern.findall(plan)
        # Also look for tool names as plain words (e.g., 'get_knowledge')
        keyword_pattern = re.compile(r'\b(get_knowledge|get_status|get_metrics|get_logs|get_ai_service|get_base_station|get_cell|get_ue|get_simulation|get_ric|get_xapp|query|search|monitor|analyze|diagnose|recommend|plan|execute|deploy|evaluate|summarize|report)\b')
        found += keyword_pattern.findall(plan)
        # Deduplicate
        return list(set(found))

    def _extract_tools_from_tools_used(self, tools_used: List[str]) -> List[str]:
        # Normalize tool names (strip, lower, etc.)
        return [t.strip().lower() for t in tools_used if t]

    def _jaccard_similarity(self, set1: set, set2: set) -> float:
        if not set1 and not set2:
            return 1.0  # Both empty = perfect match
        if not set1 or not set2:
            return 0.0
        return len(set1 & set2) / len(set1 | set2)

    async def _llm_extract_tools_and_score(self, plan: str, executed_tools: set) -> (list, float, str):
        """
        Use an LLM agent (via openai_agents_sdk) to extract intended tools from the plan and score the match.
        Returns: (extracted_tools, score, reasoning)
        """
        from agents import Runner
        import re
        # Only valid tool names
        valid_tools = {"get_knowledge", "get_knowledge_bulk"}
        # Compose prompt for the agent
        prompt = f"""
You are an expert AI system evaluator. The only valid tool names are: \"get_knowledge\" and \"get_knowledge_bulk\".
Given the following agent plan and the list of tools actually used, extract the intended tools from the plan (as a Python list of tool names, using only the valid tool names), then score the match between the plan and execution (0.0 to 1.0, where 1.0 is perfect match, 0.0 is no overlap). Provide a brief reasoning.

Plan:
{plan}

Actual tools used:
{list(executed_tools)}

Respond in JSON with keys: extracted_tools (list), score (float), reasoning (string).
"""
        agent = Agent(name="PlanEM-LLM-Evaluator")
        response = ""
        runner = Runner.run_streamed(agent, prompt)
        async for event in runner.stream_events():
            if event.type == "raw_response_event":
                delta = getattr(event.data, "delta", None)
                if isinstance(delta, str):
                    response += delta
                elif isinstance(event.data, str):
                    response += event.data
            elif event.type == "message_output_item":
                if isinstance(event.data, str):
                    response += event.data
            elif event.type == "run_completed":
                break
        # Try to extract the JSON from the response robustly
        json_match = re.search(r'\{[\s\S]*?\}', response)
        parsed = None
        extracted_tools = []
        score = None
        reasoning = response.strip()
        if json_match:
            json_str = json_match.group(0)
            try:
                parsed = ast.literal_eval(json_str)
                extracted_tools = parsed.get('extracted_tools', [])
                # Normalize/mapping: map any synonyms to valid tool names
                normalized_tools = []
                for t in extracted_tools:
                    t_lower = t.strip().lower()
                    if t_lower in valid_tools:
                        normalized_tools.append(t_lower)
                    elif t_lower in {"knowledge_query", "knowledge_query_key", "knowledge_query_bulk", "bulk_knowledge_query"}:
                        # Map to canonical names
                        if "bulk" in t_lower:
                            normalized_tools.append("get_knowledge_bulk")
                        else:
                            normalized_tools.append("get_knowledge")
                # Remove duplicates and filter to valid tools
                extracted_tools = list({tool for tool in normalized_tools if tool in valid_tools})
                score = parsed.get('score', None)
                reasoning = parsed.get('reasoning', reasoning)
            except Exception:
                # Parsing failed, keep raw response as reasoning
                extracted_tools, score = [], None
        return extracted_tools, score, reasoning

    def _plan_is_ambiguous(self, plan: str, planned_tools: set) -> bool:
        # Heuristic: ambiguous if plan is very short or no tools found
        return (not planned_tools) or (len(plan.strip()) < 30)

    def compute_plan_em_for_all(self, output_path: str = None, use_llm: bool = True) -> list:
        """
        Compute Plan.EM (Plan Execution Match) for all questions using Jaccard similarity (using expected_tools as planned_tools) and always LLM for scoring.
        Output per-question and aggregate results to JSON.
        """
        import asyncio
        results = []
        async def process_all():
            for qid, q in self.questions.items():
                # Use expected_tools from conversation_data.json as planned_tools
                planned_tools = set([t.strip().lower() for t in q.get('expected_tools', [])])
                plan_info = self.extract_user_request_and_plan(qid)
                if not plan_info:
                    continue
                plan = plan_info['decomposition_plan']
                dyn_eval = self._find_dynamic_eval_for_qid(qid)
                tools_used = []
                if dyn_eval and 'tools_used' in dyn_eval:
                    tools_used = dyn_eval['tools_used']
                executed_tools = set([t.strip().lower() for t in tools_used])
                jaccard = self._jaccard_similarity(planned_tools, executed_tools)
                # Always call LLM for every question
                print(f"Calling LLM agent for question {qid}...")
                extracted_tools, llm_score, llm_reasoning = await self._llm_extract_tools_and_score(plan, executed_tools)
                llm_tools = extracted_tools
                results.append({
                    'question_id': qid,
                    'planned_tools': list(planned_tools),
                    'executed_tools': list(executed_tools),
                    'plan_em_score': jaccard,
                    'llm_score': llm_score,
                    'llm_reasoning': llm_reasoning,
                    'llm_tools': llm_tools,
                    'plan_text': plan,
                    'tools_used': tools_used
                })
            # Aggregate
            scores = [r['plan_em_score'] for r in results if r['plan_em_score'] is not None]
            llm_scores = [r['llm_score'] for r in results if r['llm_score'] is not None]
            aggregate = {
                'mean_plan_em': sum(scores) / len(scores) if scores else None,
                'mean_llm_score': sum(llm_scores) / len(llm_scores) if llm_scores else None,
                'num_questions': len(scores),
                'scores': scores,
                'llm_scores': llm_scores
            }
            output = {
                'per_question': results,
                'aggregate': aggregate,
                'note': 'llm_score is the recommended metric for Plan.EM. plan_em_score is Jaccard similarity between expected_tools and executed_tools.'
            }
            if output_path:
                with open(output_path, 'w') as f:
                    json.dump(output, f, indent=2)
            return results
        return asyncio.run(process_all())

    def compute_act_em_for_all(self, output_path: str = None) -> list:
        """
        Compute Act.EM (Action Execution Match) for all questions using Jaccard similarity between expected_tools and tools_used.
        Output per-question and aggregate results to JSON.
        """
        results = []
        for qid, q in self.questions.items():
            planned_tools = set([t.strip().lower() for t in q.get('expected_tools', [])])
            dyn_eval = self._find_dynamic_eval_for_qid(qid)
            tools_used = []
            if dyn_eval and 'tools_used' in dyn_eval:
                tools_used = dyn_eval['tools_used']
            executed_tools = set([t.strip().lower() for t in tools_used])
            jaccard = self._jaccard_similarity(planned_tools, executed_tools)
            # Reasoning
            if not planned_tools and not executed_tools:
                reasoning = "No tools were planned or used. Perfect match."
            elif not planned_tools:
                reasoning = f"No tools were planned, but these were used: {list(executed_tools)}. Score is 0."
            elif not executed_tools:
                reasoning = f"Planned tools: {list(planned_tools)}, but none were used. Score is 0."
            else:
                intersection = planned_tools & executed_tools
                missing = planned_tools - executed_tools
                extra = executed_tools - planned_tools
                reasoning = f"Planned: {list(planned_tools)}; Used: {list(executed_tools)}; Matched: {list(intersection)}; Missing: {list(missing)}; Extra: {list(extra)}. Jaccard score: {jaccard:.2f}."
            results.append({
                'question_id': qid,
                'planned_tools': list(planned_tools),
                'executed_tools': list(executed_tools),
                'act_em_score': jaccard,
                'reasoning': reasoning,
                'tools_used': tools_used
            })
        # Aggregate
        scores = [r['act_em_score'] for r in results if r['act_em_score'] is not None]
        aggregate = {
            'mean_act_em': sum(scores) / len(scores) if scores else None,
            'num_questions': len(scores),
            'scores': scores
        }
        output = {
            'per_question': results,
            'aggregate': aggregate,
            'note': 'act_em_score is Jaccard similarity between expected_tools and tools_used.'
        }
        if output_path:
            with open(output_path, 'w') as f:
                json.dump(output, f, indent=2)
        return results

    async def _llm_detect_hallucination(self, agent_response: str, tool_outputs: dict) -> (float, str):
        from agents import Runner
        import re
        # Compose prompt for the agent
        prompt = f"""
You are an expert evaluator for AI agent tool use. Given the agent's response and the actual tool outputs, identify any hallucinations (claims or facts in the response that are not supported by the tool outputs). For each unsupported claim, explain why it is a hallucination. Then, provide a hallucination rate: the fraction of unsupported claims over total factual claims. If there are no hallucinations, the rate is 0.0. If all claims are unsupported, the rate is 1.0.

Agent Response:
{agent_response}

Tool Outputs:
{tool_outputs}

Respond in JSON with keys: hallucination_rate (float), reasoning (string).
"""
        agent = Agent(name="Hallucination-LLM-Evaluator")
        response = ""
        runner = Runner.run_streamed(agent, prompt)
        async for event in runner.stream_events():
            if event.type == "raw_response_event":
                delta = getattr(event.data, "delta", None)
                if isinstance(delta, str):
                    response += delta
                elif isinstance(event.data, str):
                    response += event.data
            elif event.type == "message_output_item":
                if isinstance(event.data, str):
                    response += event.data
            elif event.type == "run_completed":
                break
        # Try to extract the JSON from the response robustly
        json_match = re.search(r'\{[\s\S]*?\}', response)
        hallucination_rate = None
        reasoning = response.strip()
        if json_match:
            json_str = json_match.group(0)
            try:
                parsed = ast.literal_eval(json_str)
                hallucination_rate = parsed.get('hallucination_rate', None)
                reasoning = parsed.get('reasoning', reasoning)
            except Exception:
                hallucination_rate = None
        return hallucination_rate, reasoning

    def compute_hallucination_rate_for_all(self, output_path: str = None) -> list:
        """
        Compute hallucination rate for all questions using LLM to detect unsupported claims in agent responses.
        Output per-question and aggregate results to JSON.
        """
        import asyncio
        results = []
        async def process_all():
            for qid, q in self.questions.items():
                dyn_eval = self._find_dynamic_eval_for_qid(qid)
                agent_response = ""
                tool_outputs = {}
                if dyn_eval:
                    agent_response = dyn_eval.get('agent_response', '')
                    tool_outputs = dyn_eval.get('tool_outputs', {})
                print(f"Calling LLM hallucination evaluator for question {qid}...")
                hallucination_rate, reasoning = await self._llm_detect_hallucination(agent_response, tool_outputs)
                results.append({
                    'question_id': qid,
                    'hallucination_rate': hallucination_rate,
                    'reasoning': reasoning,
                    'agent_response': agent_response,
                    'tool_outputs': tool_outputs
                })
            # Aggregate
            rates = [r['hallucination_rate'] for r in results if r['hallucination_rate'] is not None]
            aggregate = {
                'mean_hallucination_rate': sum(rates) / len(rates) if rates else None,
                'num_questions': len(rates),
                'rates': rates
            }
            output = {
                'per_question': results,
                'aggregate': aggregate,
                'note': 'hallucination_rate is the fraction of unsupported claims in the agent response, as judged by LLM.'
            }
            if output_path:
                with open(output_path, 'w') as f:
                    json.dump(output, f, indent=2)
            return results
        return asyncio.run(process_all())

    def compute_tsr_for_all(self, output_path: str = None) -> list:
        """
        Compute Tool Success Rate (TSR) for all questions: fraction of tool calls that succeeded.
        Output per-question and aggregate results to JSON.
        """
        def is_successful(output):
            if output is None:
                return False
            if isinstance(output, str):
                lower = output.lower()
                if not lower.strip():
                    return False
                if 'error' in lower or 'not found' in lower or 'failed' in lower or 'exception' in lower:
                    return False
            return True

        results = []
        for qid, q in self.questions.items():
            dyn_eval = self._find_dynamic_eval_for_qid(qid)
            tools_used = []
            tool_outputs = {}
            if dyn_eval:
                tools_used = dyn_eval.get('tools_used', [])
                tool_outputs = dyn_eval.get('tool_outputs', {})
            total_calls = len(tools_used)
            success_count = 0
            call_results = []
            for i, tool in enumerate(tools_used):
                # Try to match tool output by order or by tool name
                output = None
                if isinstance(tool_outputs, dict):
                    # If only one tool type, may be a list or single output
                    if tool in tool_outputs:
                        output = tool_outputs[tool]
                    elif isinstance(tool_outputs, list) and i < len(tool_outputs):
                        output = tool_outputs[i]
                if output is None and isinstance(tool_outputs, list) and i < len(tool_outputs):
                    output = tool_outputs[i]
                if output is None:
                    output = ''
                success = is_successful(output)
                if success:
                    success_count += 1
                call_results.append({'tool': tool, 'output': output, 'success': success})
            tsr = success_count / total_calls if total_calls > 0 else None
            reasoning = f"{success_count} out of {total_calls} tool calls succeeded. "
            for cr in call_results:
                reasoning += f"[{cr['tool']}: {'success' if cr['success'] else 'fail'}] "
            results.append({
                'question_id': qid,
                'tsr': tsr,
                'reasoning': reasoning.strip(),
                'tools_used': tools_used,
                'tool_outputs': tool_outputs
            })
        # Aggregate
        tsr_scores = [r['tsr'] for r in results if r['tsr'] is not None]
        aggregate = {
            'mean_tsr': sum(tsr_scores) / len(tsr_scores) if tsr_scores else None,
            'num_questions': len(tsr_scores),
            'tsr_scores': tsr_scores
        }
        output = {
            'per_question': results,
            'aggregate': aggregate,
            'note': 'tsr is the fraction of successful tool calls per question.'
        }
        if output_path:
            with open(output_path, 'w') as f:
                json.dump(output, f, indent=2)
        return results

    def visualize_layer2_metrics(self):
        """
        Generate and save visualizations for all Layer 2 metrics (Plan.EM, Act.EM, Hallucination Rate, TSR).
        Plots are saved in hatt_e/visualization/ with clear filenames and _hatt_e_l2.png suffix.
        """
        import matplotlib.pyplot as plt
        import os
        import json
        import numpy as np

        # Save to hatt_e/visualization/ (not layer2/visualization)
        vis_dir = os.path.join(os.path.dirname(self.output_dir), 'visualization')
        os.makedirs(vis_dir, exist_ok=True)

        # Helper to load results
        def load_json(path):
            if os.path.exists(path):
                with open(path, 'r') as f:
                    return json.load(f)
            return None

        # Plan.EM
        planem_path = os.path.join(self.output_dir.replace('layer1', 'layer2'), 'plan_em_results.json')
        planem = load_json(planem_path)
        if planem:
            llm_scores = [r['llm_score'] for r in planem['per_question'] if r['llm_score'] is not None]
            jaccard_scores = [r['plan_em_score'] for r in planem['per_question'] if r['plan_em_score'] is not None]
            plt.figure()
            plt.hist(llm_scores, bins=np.arange(0, 1.05, 0.05), color='skyblue', edgecolor='black')
            plt.title('Plan.EM LLM Score Distribution (Layer 2)')
            plt.xlabel('LLM Score')
            plt.ylabel('Count')
            plt.savefig(os.path.join(vis_dir, 'plan_em_llm_histogram_hatt_e_l2.png'))
            plt.close()
            plt.figure()
            plt.hist(jaccard_scores, bins=np.arange(0, 1.05, 0.05), color='orange', edgecolor='black')
            plt.title('Plan.EM Jaccard Score Distribution (Layer 2)')
            plt.xlabel('Jaccard Score')
            plt.ylabel('Count')
            plt.savefig(os.path.join(vis_dir, 'plan_em_jaccard_histogram_hatt_e_l2.png'))
            plt.close()

        # Act.EM
        actem_path = os.path.join(self.output_dir.replace('layer1', 'layer2'), 'act_em_results.json')
        actem = load_json(actem_path)
        if actem:
            actem_scores = [r['act_em_score'] for r in actem['per_question'] if r['act_em_score'] is not None]
            plt.figure()
            plt.hist(actem_scores, bins=np.arange(0, 1.05, 0.05), color='green', edgecolor='black')
            plt.title('Act.EM Score Distribution (Layer 2)')
            plt.xlabel('Act.EM Score')
            plt.ylabel('Count')
            plt.savefig(os.path.join(vis_dir, 'act_em_histogram_hatt_e_l2.png'))
            plt.close()

        # Hallucination Rate
        halluc_path = os.path.join(self.output_dir.replace('layer1', 'layer2'), 'hallucination_results.json')
        halluc = load_json(halluc_path)
        if halluc:
            halluc_scores = [r['hallucination_rate'] for r in halluc['per_question'] if r['hallucination_rate'] is not None]
            plt.figure()
            plt.hist(halluc_scores, bins=np.arange(0, 1.05, 0.05), color='red', edgecolor='black')
            plt.title('Hallucination Rate Distribution (Layer 2)')
            plt.xlabel('Hallucination Rate')
            plt.ylabel('Count')
            plt.savefig(os.path.join(vis_dir, 'hallucination_rate_histogram_hatt_e_l2.png'))
            plt.close()

        # TSR
        tsr_path = os.path.join(self.output_dir.replace('layer1', 'layer2'), 'tsr_results.json')
        tsr = load_json(tsr_path)
        if tsr:
            tsr_scores = [r['tsr'] for r in tsr['per_question'] if r['tsr'] is not None]
            plt.figure()
            plt.hist(tsr_scores, bins=np.arange(0, 1.05, 0.05), color='purple', edgecolor='black')
            plt.title('Tool Success Rate (TSR) Distribution (Layer 2)')
            plt.xlabel('TSR')
            plt.ylabel('Count')
            plt.savefig(os.path.join(vis_dir, 'tsr_histogram_hatt_e_l2.png'))
            plt.close()

        print(f"[Layer 2 Visualizations] Saved to {vis_dir}")

    async def compute_layer3_tsr_llm(self, output_path: str = None):
        """
        Compute Task Success Rate (TSR) for Layer 3 using LLM judge (OpenAI Agents SDK).
        For each question, ask the LLM if the final answer is correct and complete for the user request.
        Output per-question and aggregate results to JSON.
        """
        from agents import Agent, Runner
        
        print(f"[Layer 3 TSR] Starting evaluation for {len(self.questions)} questions...")
        results = []
        total_questions = len(self.questions)
        
        for i, (qid, q) in enumerate(self.questions.items(), 1):
            print(f"[Layer 3 TSR] Processing question {i}/{total_questions}: {qid}")
            
            dyn_eval = self._find_dynamic_eval_for_qid(qid)
            if not dyn_eval:
                print(f"[Layer 3 TSR] ⚠️  No dynamic evaluation found for {qid}, skipping...")
                continue
                
            user_question = q.get('static_question', '')
            reference_answer = q.get('reference_answer', '') if 'reference_answer' in q else ''
            agent_response = dyn_eval.get('agent_response', '')
            
            if not agent_response:
                print(f"[Layer 3 TSR] ⚠️  No agent response found for {qid}, skipping...")
                continue
                
            print(f"[Layer 3 TSR] 📝 User question: {user_question[:100]}{'...' if len(user_question) > 100 else ''}")
            print(f"[Layer 3 TSR] 🤖 Agent response length: {len(agent_response)} characters")
            if reference_answer:
                print(f"[Layer 3 TSR] 📚 Reference answer available: {len(reference_answer)} characters")
            else:
                print(f"[Layer 3 TSR] 📚 No reference answer available")
            
            # Compose prompt
            prompt = f"""
You are an expert evaluator for AI system task success. Given the original user request, the system's final answer, and (if available) a reference answer, judge if the system's answer is correct and complete.

User Request:
{user_question}

System Final Answer:
{agent_response}

Reference Answer (if available):
{reference_answer}

Strictly follow these instructions:
- Output SCORE: 1 if the answer is correct and complete, 0 if it is not (no partial credit).
- Output REASONING: a detailed explanation for your score.
- Output FEEDBACK: specific feedback on what was good or missing.
- Be strict: only give 1 if the answer fully and correctly addresses the user request.
- Use the reference answer for comparison if provided, but do not require exact wording.

Respond in this exact format:
SCORE: <0 or 1>
REASONING: <your reasoning>
FEEDBACK: <your feedback>
"""
            print(f"[Layer 3 TSR] 🤖 Calling LLM judge for {qid}...")
            
            agent = Agent(name="Layer3-TSR-LLM-Evaluator")
            response = ""
            runner = Runner.run_streamed(agent, prompt)
            
            async for event in runner.stream_events():
                if event.type == "raw_response_event":
                    delta = getattr(event.data, "delta", None)
                    if isinstance(delta, str):
                        response += delta
                    elif isinstance(event.data, str):
                        response += event.data
                elif event.type == "message_output_item":
                    if isinstance(event.data, str):
                        response += event.data
                elif event.type == "run_completed":
                    break
            
            print(f"[Layer 3 TSR] ✅ LLM response received for {qid} ({len(response)} characters)")
            
            # Parse SCORE
            score = 0
            reasoning = response.strip()
            feedback = ""
            
            for line in response.split('\n'):
                if line.strip().lower().startswith("score:"):
                    try:
                        score_val = int(re.findall(r'\d+', line)[0])
                        score = 1 if score_val == 1 else 0
                    except Exception:
                        score = 0
                elif line.strip().lower().startswith("reasoning:"):
                    reasoning = line.strip()[len("reasoning:"):].strip()
                elif line.strip().lower().startswith("feedback:"):
                    feedback = line.strip()[len("feedback:"):].strip()
            
            print(f"[Layer 3 TSR] 📊 Score for {qid}: {score} ({'✅ PASS' if score == 1 else '❌ FAIL'})")
            if reasoning:
                print(f"[Layer 3 TSR] 💭 Reasoning: {reasoning[:100]}{'...' if len(reasoning) > 100 else ''}")
            
            results.append({
                'question_id': qid,
                'user_request': user_question,
                'agent_response': agent_response,
                'reference_answer': reference_answer,
                'tsr': score,
                'reasoning': reasoning,
                'feedback': feedback,
                'llm_raw_response': response.strip()
            })
        
        print(f"[Layer 3 TSR] 📈 Processing complete! Evaluated {len(results)} questions")
        
        # Aggregate
        tsr_scores = [r['tsr'] for r in results]
        mean_tsr = sum(tsr_scores) / len(tsr_scores) if tsr_scores else 0
        pass_count = sum(tsr_scores)
        total_count = len(tsr_scores)
        
        print(f"[Layer 3 TSR] 📊 Final Results:")
        print(f"[Layer 3 TSR]   • Total questions evaluated: {total_count}")
        print(f"[Layer 3 TSR]   • Passed (score=1): {pass_count}")
        print(f"[Layer 3 TSR]   • Failed (score=0): {total_count - pass_count}")
        print(f"[Layer 3 TSR]   • Mean TSR: {mean_tsr:.3f} ({mean_tsr*100:.1f}%)")
        
        aggregate = {
            'mean_tsr': mean_tsr,
            'num_questions': total_count,
            'tsr_scores': tsr_scores,
            'pass_count': pass_count,
            'fail_count': total_count - pass_count
        }
        
        output = {
            'per_question': results,
            'aggregate': aggregate,
            'note': 'Layer 3 TSR is the fraction of questions judged as fully correct and complete by LLM.'
        }
        
        if output_path:
            print(f"[Layer 3 TSR] 💾 Saving results to {output_path}")
            with open(output_path, 'w') as f:
                json.dump(output, f, indent=2)
            print(f"[Layer 3 TSR] ✅ Results saved successfully!")
        
        return results

    async def compute_layer3_response_quality(self, output_path: str = None):
        """
        Compute Response Quality (RQ) for Layer 3 using LLM judge (OpenAI Agents SDK).
        For each question, ask the LLM to rate the quality of the final answer on a 1-5 scale.
        Output per-question and aggregate results to JSON.
        """
        from agents import Agent, Runner
        
        print(f"[Layer 3 RQ] Starting Response Quality evaluation for {len(self.questions)} questions...")
        results = []
        total_questions = len(self.questions)
        
        for i, (qid, q) in enumerate(self.questions.items(), 1):
            print(f"[Layer 3 RQ] Processing question {i}/{total_questions}: {qid}")
            
            dyn_eval = self._find_dynamic_eval_for_qid(qid)
            if not dyn_eval:
                print(f"[Layer 3 RQ] ⚠️  No dynamic evaluation found for {qid}, skipping...")
                continue
                
            user_question = q.get('static_question', '')
            reference_answer = q.get('reference_answer', '') if 'reference_answer' in q else ''
            agent_response = dyn_eval.get('agent_response', '')
            
            if not agent_response:
                print(f"[Layer 3 RQ] ⚠️  No agent response found for {qid}, skipping...")
                continue
                
            print(f"[Layer 3 RQ] 📝 User question: {user_question[:100]}{'...' if len(user_question) > 100 else ''}")
            print(f"[Layer 3 RQ] 🤖 Agent response length: {len(agent_response)} characters")
            if reference_answer:
                print(f"[Layer 3 RQ] 📚 Reference answer available: {len(reference_answer)} characters")
            else:
                print(f"[Layer 3 RQ] 📚 No reference answer available")
            
            # Compose prompt for quality assessment
            prompt = f"""
You are an expert evaluator for AI system response quality. Given the original user request and the system's final answer, rate the quality of the response on a scale of 1-5.

Quality Criteria:
- **5 (Excellent)**: Complete, accurate, well-structured, directly addresses the question, provides valuable insights
- **4 (Good)**: Mostly complete and accurate, clear structure, addresses the question well
- **3 (Fair)**: Partially complete, some accuracy issues, basic structure, partially addresses the question
- **2 (Poor)**: Incomplete, significant accuracy issues, poor structure, doesn't address the question well
- **1 (Very Poor)**: Incomplete, inaccurate, confusing, doesn't address the question

User Request:
{user_question}

System Final Answer:
{agent_response}

Reference Answer (if available):
{reference_answer}

Strictly follow these instructions:
- Output SCORE: a number from 1 to 5 based on the quality criteria above
- Output REASONING: detailed explanation for your quality score
- Output FEEDBACK: specific feedback on what was good or needs improvement
- Consider accuracy, completeness, clarity, relevance, and usefulness
- Use the reference answer for comparison if provided, but evaluate quality independently

Respond in this exact format:
SCORE: <1-5>
REASONING: <your reasoning>
FEEDBACK: <your feedback>
"""
            print(f"[Layer 3 RQ] 🤖 Calling LLM judge for quality assessment of {qid}...")
            
            agent = Agent(name="Layer3-RQ-LLM-Evaluator")
            response = ""
            runner = Runner.run_streamed(agent, prompt)
            
            async for event in runner.stream_events():
                if event.type == "raw_response_event":
                    delta = getattr(event.data, "delta", None)
                    if isinstance(delta, str):
                        response += delta
                    elif isinstance(event.data, str):
                        response += event.data
                elif event.type == "message_output_item":
                    if isinstance(event.data, str):
                        response += event.data
                elif event.type == "run_completed":
                    break
            
            print(f"[Layer 3 RQ] ✅ LLM response received for {qid} ({len(response)} characters)")
            
            # Parse SCORE
            quality_score = 3  # Default to fair
            reasoning = response.strip()
            feedback = ""
            
            for line in response.split('\n'):
                if line.strip().lower().startswith("score:"):
                    try:
                        score_val = int(re.findall(r'\d+', line)[0])
                        quality_score = max(1, min(5, score_val))  # Clamp to 1-5 range
                    except Exception:
                        quality_score = 3
                elif line.strip().lower().startswith("reasoning:"):
                    reasoning = line.strip()[len("reasoning:"):].strip()
                elif line.strip().lower().startswith("feedback:"):
                    feedback = line.strip()[len("feedback:"):].strip()
            
            # Quality level mapping
            quality_levels = {5: "Excellent", 4: "Good", 3: "Fair", 2: "Poor", 1: "Very Poor"}
            quality_level = quality_levels.get(quality_score, "Fair")
            
            print(f"[Layer 3 RQ] 📊 Quality score for {qid}: {quality_score}/5 ({quality_level})")
            if reasoning:
                print(f"[Layer 3 RQ] 💭 Reasoning: {reasoning[:100]}{'...' if len(reasoning) > 100 else ''}")
            
            results.append({
                'question_id': qid,
                'user_request': user_question,
                'agent_response': agent_response,
                'reference_answer': reference_answer,
                'quality_score': quality_score,
                'quality_level': quality_level,
                'reasoning': reasoning,
                'feedback': feedback,
                'llm_raw_response': response.strip()
            })
        
        print(f"[Layer 3 RQ] 📈 Processing complete! Evaluated {len(results)} questions")
        
        # Aggregate statistics
        quality_scores = [r['quality_score'] for r in results]
        mean_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 3.0
        
        # Quality distribution
        quality_distribution = {}
        for score in range(1, 6):
            count = quality_scores.count(score)
            quality_distribution[f"score_{score}"] = count
            quality_distribution[f"score_{score}_percentage"] = (count / len(quality_scores) * 100) if quality_scores else 0
        
        # Quality level distribution
        level_distribution = {}
        for result in results:
            level = result['quality_level']
            level_distribution[level] = level_distribution.get(level, 0) + 1
        
        print(f"[Layer 3 RQ] 📊 Final Results:")
        print(f"[Layer 3 RQ]   • Total questions evaluated: {len(quality_scores)}")
        print(f"[Layer 3 RQ]   • Mean quality score: {mean_quality:.2f}/5")
        print(f"[Layer 3 RQ]   • Quality distribution:")
        for score in range(1, 6):
            count = quality_scores.count(score)
            percentage = (count / len(quality_scores) * 100) if quality_scores else 0
            print(f"[Layer 3 RQ]     - Score {score}: {count} questions ({percentage:.1f}%)")
        
        aggregate = {
            'mean_quality_score': mean_quality,
            'num_questions': len(quality_scores),
            'quality_scores': quality_scores,
            'quality_distribution': quality_distribution,
            'level_distribution': level_distribution
        }
        
        output = {
            'per_question': results,
            'aggregate': aggregate,
            'note': 'Layer 3 RQ is the quality assessment of responses on a 1-5 scale by LLM judge.'
        }
        
        if output_path:
            print(f"[Layer 3 RQ] 💾 Saving results to {output_path}")
            with open(output_path, 'w') as f:
                json.dump(output, f, indent=2)
            print(f"[Layer 3 RQ] ✅ Results saved successfully!")
        
        return results

    async def compute_layer3_consistency(self, output_path: str = None):
        """
        Compute Consistency metrics for Layer 3 using LLM judge (OpenAI Agents SDK).
        For each question, compare responses to similar questions and evaluate consistency.
        Uses conversation logs to extract responses and limits context to avoid token limits.
        Output per-question and aggregate results to JSON.
        """
        from agents import Agent, Runner
        
        print(f"[Layer 3 Consistency] Starting consistency evaluation for {len(self.questions)} questions...")
        results = []
        total_questions = len(self.questions)
        
        # Load conversation logs
        conversation_logs = self._load_conversation_logs()
        if not conversation_logs:
            print(f"[Layer 3 Consistency] ⚠️  No conversation logs found, cannot proceed")
            return results
        
        # Group questions by similarity (based on keywords and topics)
        question_groups = self._group_questions_by_similarity()
        print(f"[Layer 3 Consistency] Identified {len(question_groups)} question groups for consistency analysis")
        
        for i, (qid, q) in enumerate(self.questions.items(), 1):
            print(f"[Layer 3 Consistency] Processing question {i}/{total_questions}: {qid}")
            
            # Extract response from conversation logs
            current_response = self._extract_response_from_logs(qid, conversation_logs)
            if not current_response:
                print(f"[Layer 3 Consistency] ⚠️  No response found for {qid} in conversation logs, skipping...")
                continue
                
            user_question = q.get('static_question', '')
            
            # Find similar questions in the same group
            similar_questions = self._find_similar_questions(qid, question_groups)
            
            if not similar_questions:
                print(f"[Layer 3 Consistency] ⚠️  No similar questions found for {qid}, assigning default consistency score")
                consistency_score = 3.0  # Default to neutral
                reasoning = "No similar questions available for comparison"
                feedback = "Cannot evaluate consistency without similar questions"
            else:
                print(f"[Layer 3 Consistency] 🔍 Found {len(similar_questions)} similar questions for comparison")
                
                # Get responses for similar questions (limit to 3 to avoid token limits)
                similar_responses = []
                for similar_qid in similar_questions[:3]:  # Limit to 3 similar questions
                    similar_response = self._extract_response_from_logs(similar_qid, conversation_logs)
                    if similar_response:
                        similar_responses.append({
                            'question_id': similar_qid,
                            'question': self.questions[similar_qid].get('static_question', ''),
                            'response': similar_response
                        })
                
                if not similar_responses:
                    print(f"[Layer 3 Consistency] ⚠️  No valid responses found for similar questions")
                    consistency_score = 3.0
                    reasoning = "No valid responses from similar questions for comparison"
                    feedback = "Cannot evaluate consistency without comparable responses"
                else:
                    print(f"[Layer 3 Consistency] 📝 Using {len(similar_responses)} similar responses for comparison")
                    # Use LLM to evaluate consistency
                    consistency_score, reasoning, feedback = await self._evaluate_consistency_llm(
                        user_question, current_response, similar_responses
                    )
            
            print(f"[Layer 3 Consistency] 📊 Consistency score for {qid}: {consistency_score:.2f}/5")
            if reasoning:
                print(f"[Layer 3 Consistency] 💭 Reasoning: {reasoning[:100]}{'...' if len(reasoning) > 100 else ''}")
            
            results.append({
                'question_id': qid,
                'user_request': user_question,
                'agent_response': current_response,
                'similar_questions': [r['question_id'] for r in similar_responses] if 'similar_responses' in locals() else [],
                'consistency_score': consistency_score,
                'reasoning': reasoning,
                'feedback': feedback,
                'num_similar_questions': len(similar_responses) if 'similar_responses' in locals() else 0
            })
        
        print(f"[Layer 3 Consistency] 📈 Processing complete! Evaluated {len(results)} questions")
        
        # Aggregate statistics
        consistency_scores = [r['consistency_score'] for r in results]
        mean_consistency = sum(consistency_scores) / len(consistency_scores) if consistency_scores else 3.0
        
        # Consistency distribution
        consistency_distribution = {}
        for score_range in [(1, 2), (2, 3), (3, 4), (4, 5)]:
            count = len([s for s in consistency_scores if score_range[0] <= s < score_range[1]])
            consistency_distribution[f"score_{score_range[0]}_{score_range[1]}"] = count
            consistency_distribution[f"score_{score_range[0]}_{score_range[1]}_percentage"] = (count / len(consistency_scores) * 100) if consistency_scores else 0
        
        print(f"[Layer 3 Consistency] 📊 Final Results:")
        print(f"[Layer 3 Consistency]   • Total questions evaluated: {len(consistency_scores)}")
        print(f"[Layer 3 Consistency]   • Mean consistency score: {mean_consistency:.2f}/5")
        print(f"[Layer 3 Consistency]   • Consistency distribution:")
        for score_range in [(1, 2), (2, 3), (3, 4), (4, 5)]:
            count = len([s for s in consistency_scores if score_range[0] <= s < score_range[1]])
            percentage = (count / len(consistency_scores) * 100) if consistency_scores else 0
            print(f"[Layer 3 Consistency]     - Score {score_range[0]}-{score_range[1]}: {count} questions ({percentage:.1f}%)")
        
        aggregate = {
            'mean_consistency_score': mean_consistency,
            'num_questions': len(consistency_scores),
            'consistency_scores': consistency_scores,
            'consistency_distribution': consistency_distribution
        }
        
        output = {
            'per_question': results,
            'aggregate': aggregate,
            'note': 'Layer 3 Consistency evaluates how consistent responses are across similar questions using LLM judge.'
        }
        
        if output_path:
            print(f"[Layer 3 Consistency] 💾 Saving results to {output_path}")
            with open(output_path, 'w') as f:
                json.dump(output, f, indent=2)
            print(f"[Layer 3 Consistency] ✅ Results saved successfully!")
        
        return results

    def _load_conversation_logs(self):
        """
        Load conversation logs from the all_conversation_logs file.
        Returns a dictionary mapping question_id to conversation log content.
        """
        # Dynamically find the conversation logs file
        logs_file = None
        for file in os.listdir(self.results_dir):
            if file.startswith("all_conversation_logs_") and file.endswith(".txt"):
                logs_file = os.path.join(self.results_dir, file)
                break
        
        if not logs_file or not os.path.exists(logs_file):
            print(f"[Layer 3 Consistency] ⚠️  Conversation logs file not found in {self.results_dir}")
            print(f"[Layer 3 Consistency] Looking for files starting with 'all_conversation_logs_'")
            return {}
        
        try:
            with open(logs_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse conversation logs by question
            logs = {}
            current_qid = None
            current_log = []
            
            for line in content.split('\n'):
                if line.startswith('=== CONVERSATION LOG FOR '):
                    if current_qid and current_log:
                        logs[current_qid] = '\n'.join(current_log)
                    current_qid = line.split('FOR ')[1].split(' ===')[0]
                    current_log = [line]
                elif line.startswith('=== END LOG ==='):
                    current_log.append(line)
                    if current_qid:
                        logs[current_qid] = '\n'.join(current_log)
                        current_qid = None
                        current_log = []
                elif current_qid:
                    current_log.append(line)
            
            print(f"[Layer 3 Consistency] 📚 Loaded conversation logs for {len(logs)} questions")
            return logs
            
        except Exception as e:
            print(f"[Layer 3 Consistency] ❌ Error loading conversation logs: {e}")
            return {}

    def _extract_response_from_logs(self, qid, conversation_logs):
        """
        Extract the agent response from conversation logs for a specific question.
        """
        if qid not in conversation_logs:
            return None
        
        log_content = conversation_logs[qid]
        
        # Look for AGENT RESPONSE section
        if 'AGENT RESPONSE:' in log_content:
            # Extract everything after AGENT RESPONSE: until the next section
            start_idx = log_content.find('AGENT RESPONSE:') + len('AGENT RESPONSE:')
            end_idx = log_content.find('\n\n', start_idx)
            if end_idx == -1:
                end_idx = log_content.find('TOOLS USED:', start_idx)
            if end_idx == -1:
                end_idx = len(log_content)
            
            response = log_content[start_idx:end_idx].strip()
            return response
        
        return None

    def _group_questions_by_similarity(self):
        """
        Group questions by similarity based on keywords and topics.
        Returns a dictionary mapping group_id to list of question_ids.
        """
        groups = {}
        group_keywords = {
            'network_status': ['status', 'current', 'network', 'base station', 'cell'],
            'ai_services': ['ai', 'artificial intelligence', 'service', 'monitor', 'xapp'],
            'performance': ['performance', 'optimization', 'bottleneck', 'efficiency'],
            'user_equipment': ['user equipment', 'ue', 'imsi', 'mobile', 'device'],
            'ric_operations': ['ric', 'ran intelligent controller', 'xapp', 'controller'],
            'network_config': ['configuration', 'setup', 'deployment', 'infrastructure'],
            'mobility': ['handover', 'roaming', 'mobility', 'movement', 'cell selection'],
            'technical_details': ['technical', 'specification', 'parameter', 'detail']
        }
        
        for qid, q in self.questions.items():
            question_text = q.get('static_question', '').lower()
            assigned_group = None
            
            for group_name, keywords in group_keywords.items():
                if any(keyword in question_text for keyword in keywords):
                    assigned_group = group_name
                    break
            
            if not assigned_group:
                assigned_group = 'general'
            
            if assigned_group not in groups:
                groups[assigned_group] = []
            groups[assigned_group].append(qid)
        
        return groups

    def _find_similar_questions(self, qid, question_groups):
        """
        Find similar questions for a given question_id.
        Returns list of similar question_ids.
        """
        for group_name, group_questions in question_groups.items():
            if qid in group_questions:
                # Return other questions in the same group (excluding the current one)
                return [q for q in group_questions if q != qid]
        return []

    async def _evaluate_consistency_llm(self, current_question, current_response, similar_responses):
        """
        Use LLM to evaluate consistency between current response and similar responses.
        Returns (consistency_score, reasoning, feedback).
        Limits context to avoid token limits and includes better error handling.
        """
        from agents import Agent, Runner
        
        # Prepare similar responses text (truncate long responses to avoid token limits)
        similar_text = ""
        for i, resp in enumerate(similar_responses, 1):
            # Truncate responses to ~400 characters to stay well within token limits
            truncated_response = resp['response'][:400] + "..." if len(resp['response']) > 400 else resp['response']
            similar_text += f"\nSimilar Question {i} ({resp['question_id']}):\n{resp['question'][:150]}...\nResponse: {truncated_response}\n"
        
        # Estimate token count and truncate if needed
        estimated_tokens = len(current_question + current_response + similar_text) / 4  # Rough estimate
        if estimated_tokens > 20000:  # More conservative limit
            print(f"[Layer 3 Consistency] ⚠️  Estimated tokens too high ({estimated_tokens:.0f}), truncating responses")
            # Truncate current response
            current_response = current_response[:800] + "..." if len(current_response) > 800 else current_response
            # Further truncate similar responses
            similar_text = ""
            for i, resp in enumerate(similar_responses, 1):
                truncated_response = resp['response'][:250] + "..." if len(resp['response']) > 250 else resp['response']
                similar_text += f"\nSimilar Question {i} ({resp['question_id']}):\n{resp['question'][:100]}...\nResponse: {truncated_response}\n"
        
        prompt = f"""
You are an expert evaluator for AI system response consistency. Given a current question and response, along with responses to similar questions, evaluate how consistent the current response is with the similar responses.

Consistency Criteria:
- **5 (Highly Consistent)**: Response follows same patterns, style, depth, and approach as similar responses
- **4 (Mostly Consistent)**: Response is generally consistent with minor variations in style or depth
- **3 (Moderately Consistent)**: Response has some consistency but notable differences in approach or detail level
- **2 (Inconsistent)**: Response differs significantly from similar responses in style, depth, or approach
- **1 (Highly Inconsistent)**: Response is completely different from similar responses

Current Question: {current_question}
Current Response: {current_response}

Similar Questions and Responses:{similar_text}

Strictly follow these instructions:
- Output SCORE: a number from 1 to 5 based on the consistency criteria above
- Output REASONING: detailed explanation for your consistency score
- Output FEEDBACK: specific feedback on consistency strengths and weaknesses
- Consider style, depth, approach, terminology, and overall response patterns
- Evaluate whether the response maintains similar quality and structure as similar responses

Respond in this exact format:
SCORE: <1-5>
REASONING: <your reasoning>
FEEDBACK: <your feedback>
"""
        
        try:
            agent = Agent(name="Layer3-Consistency-LLM-Evaluator")
            response = ""
            runner = Runner.run_streamed(agent, prompt)
            
            async for event in runner.stream_events():
                if event.type == "raw_response_event":
                    delta = getattr(event.data, "delta", None)
                    if isinstance(delta, str):
                        response += delta
                    elif isinstance(event.data, str):
                        response += event.data
                elif event.type == "message_output_item":
                    if isinstance(event.data, str):
                        response += event.data
                elif event.type == "run_completed":
                    break
            
            # Validate response quality
            if not response or len(response.strip()) < 50:
                print(f"[Layer 3 Consistency] ⚠️  LLM response too short or empty")
                return 3.0, "LLM response was too short or empty", "Unable to evaluate consistency due to insufficient LLM response"
            
            # Check for error messages in response
            error_indicators = [
                "i'm sorry", "i can't assist", "i cannot help", "unable to", "error", 
                "cannot process", "not available", "sorry, but", "i apologize"
            ]
            
            response_lower = response.lower()
            if any(indicator in response_lower for indicator in error_indicators):
                print(f"[Layer 3 Consistency] ⚠️  LLM returned error message: {response[:100]}...")
                return 3.0, f"LLM returned error: {response.strip()}", "Unable to evaluate consistency due to LLM error"
            
            # Parse SCORE
            consistency_score = 3.0  # Default to moderate
            reasoning = response.strip()
            feedback = ""
            
            for line in response.split('\n'):
                if line.strip().lower().startswith("score:"):
                    try:
                        score_val = int(re.findall(r'\d+', line)[0])
                        consistency_score = max(1, min(5, score_val))  # Clamp to 1-5 range
                    except Exception:
                        consistency_score = 3.0
                elif line.strip().lower().startswith("reasoning:"):
                    reasoning = line.strip()[len("reasoning:"):].strip()
                elif line.strip().lower().startswith("feedback:"):
                    feedback = line.strip()[len("feedback:"):].strip()
            
            # Validate that we got meaningful reasoning
            if not reasoning or reasoning.strip() == response.strip():
                print(f"[Layer 3 Consistency] ⚠️  No meaningful reasoning extracted")
                return 3.0, "No meaningful reasoning could be extracted from LLM response", "Unable to parse LLM response properly"
            
            return consistency_score, reasoning, feedback
            
        except Exception as e:
            print(f"[Layer 3 Consistency] ❌ Error during LLM evaluation: {e}")
            return 3.0, f"Error during LLM evaluation: {str(e)}", "LLM evaluation failed due to technical error"

    async def compute_layer3_system_cost(self, output_path: str = None):
        """
        Compute System Cost metrics for Layer 3.
        Extract cost metrics from evaluation data including API calls, token usage, and processing time.
        Output per-question and aggregate results to JSON.
        """
        print(f"[Layer 3 System Cost] Starting system cost evaluation for {len(self.questions)} questions...")
        results = []
        total_questions = len(self.questions)

        # Load dynamic evaluation data for cost metrics
        dyn_eval = self._load_dynamic_eval()
        if not dyn_eval:
            print(f"[Layer 3 System Cost] ⚠️  No dynamic evaluation data found, cannot proceed")
            return results

        for i, (qid, q) in enumerate(self.questions.items(), 1):
            print(f"[Layer 3 System Cost] Processing question {i}/{total_questions}: {qid}")

            # Find evaluation data for this question
            eval_data = self._find_dynamic_eval_for_qid(qid)
            if not eval_data:
                print(f"[Layer 3 System Cost] ⚠️  No evaluation data found for {qid}, skipping...")
                continue

            user_question = q.get('static_question', '')
            
            # Extract cost metrics
            cost_metrics = self._extract_cost_metrics(eval_data)
            
            # Calculate total cost
            total_cost = self._calculate_total_cost(cost_metrics)
            
            # Determine cost efficiency score (1-5 scale)
            cost_efficiency = self._calculate_cost_efficiency(cost_metrics, total_cost)
            
            result = {
                "question_id": qid,
                "user_request": user_question,
                "cost_metrics": cost_metrics,
                "total_cost": total_cost,
                "cost_efficiency_score": cost_efficiency,
                "cost_breakdown": {
                    "api_calls": cost_metrics.get('api_calls', 0),
                    "estimated_tokens": cost_metrics.get('estimated_tokens', 0),
                    "processing_time_ms": cost_metrics.get('processing_time_ms', 0),
                    "tool_calls": cost_metrics.get('tool_calls', 0)
                }
            }
            
            results.append(result)
            print(f"[Layer 3 System Cost] ✅ Processed {qid} - Total Cost: {total_cost:.4f}, Efficiency: {cost_efficiency}/5")

        # Calculate aggregate statistics
        if results:
            total_costs = [r['total_cost'] for r in results]
            efficiency_scores = [r['cost_efficiency_score'] for r in results]
            
            aggregate = {
                "mean_total_cost": sum(total_costs) / len(total_costs),
                "median_total_cost": sorted(total_costs)[len(total_costs)//2],
                "min_total_cost": min(total_costs),
                "max_total_cost": max(total_costs),
                "mean_cost_efficiency": sum(efficiency_scores) / len(efficiency_scores),
                "total_questions": len(results),
                "cost_distribution": {
                    "low_cost": len([c for c in total_costs if c < 0.01]),
                    "medium_cost": len([c for c in total_costs if 0.01 <= c < 0.05]),
                    "high_cost": len([c for c in total_costs if c >= 0.05])
                }
            }
            
            output = {
                "per_question": results,
                "aggregate": aggregate,
                "note": "Layer 3 System Cost evaluates the computational and financial cost of processing each question."
            }
            
            # Save results
            if output_path is None:
                output_path = os.path.join(self.results_dir, "hatt_e", "layer3", "system_cost.json")
            
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(output, f, indent=2)
            
            print(f"[Layer 3 System Cost] 💾 Results saved to {output_path}")
            print(f"[Layer 3 System Cost] 📊 Aggregate - Mean Cost: {aggregate['mean_total_cost']:.4f}, Mean Efficiency: {aggregate['mean_cost_efficiency']:.2f}/5")
        
        return results

    def _extract_cost_metrics(self, eval_data):
        """
        Extract cost-related metrics from evaluation data.
        """
        cost_metrics = {
            'api_calls': 0,
            'estimated_tokens': 0,
            'processing_time_ms': 0,
            'tool_calls': 0
        }
        
        # Extract tool calls
        if 'tools_used' in eval_data:
            cost_metrics['tool_calls'] = len(eval_data['tools_used'])
        
        # Extract processing time
        if 'response_time' in eval_data:
            cost_metrics['processing_time_ms'] = eval_data['response_time'] * 1000  # Convert to milliseconds
        
        # Estimate API calls and tokens from agent response
        if 'agent_response' in eval_data:
            response = eval_data['agent_response']
            
            # Estimate tokens (rough calculation)
            cost_metrics['estimated_tokens'] = len(response.split()) * 1.3  # Rough token estimation
            
            # Estimate API calls based on response complexity and tool usage
            base_api_calls = 1  # Base API call for the response
            tool_api_calls = cost_metrics['tool_calls']  # Each tool call is an API call
            
            if len(response) > 1000:
                base_api_calls = 3  # Complex response
            elif len(response) > 500:
                base_api_calls = 2  # Medium response
            
            cost_metrics['api_calls'] = base_api_calls + tool_api_calls
        
        return cost_metrics

    def _calculate_total_cost(self, cost_metrics):
        """
        Calculate total cost based on metrics.
        Uses estimated costs for API calls, tokens, and processing time.
        """
        # Cost estimates (in USD)
        API_CALL_COST = 0.002  # $0.002 per API call
        TOKEN_COST = 0.000002  # $0.000002 per token
        PROCESSING_COST_PER_MS = 0.000001  # $0.000001 per millisecond
        
        total_cost = (
            cost_metrics.get('api_calls', 0) * API_CALL_COST +
            cost_metrics.get('estimated_tokens', 0) * TOKEN_COST +
            cost_metrics.get('processing_time_ms', 0) * PROCESSING_COST_PER_MS
        )
        
        return total_cost

    def _calculate_cost_efficiency(self, cost_metrics, total_cost):
        """
        Calculate cost efficiency score (1-5) based on cost vs. response quality.
        """
        # Base efficiency on cost and response quality
        response_length = cost_metrics.get('estimated_tokens', 0) / 1.3  # Convert back to words
        
        if total_cost == 0:
            return 5.0  # Perfect efficiency if no cost
        
        # Efficiency factors
        cost_factor = max(0, 5 - (total_cost * 100))  # Lower cost = higher score
        quality_factor = min(5, response_length / 50)  # Longer response = higher quality
        
        # Tool usage efficiency
        tool_efficiency = min(5, 5 - cost_metrics.get('tool_calls', 0) * 0.5)  # Fewer tools = more efficient
        
        # Calculate weighted average
        efficiency_score = (cost_factor * 0.4 + quality_factor * 0.4 + tool_efficiency * 0.2)
        
        return max(1.0, min(5.0, efficiency_score))  # Clamp between 1-5

    async def compute_layer3_latency(self, output_path: str = None):
        """
        Compute Latency metrics for Layer 3.
        Extract response time metrics from evaluation data and calculate latency statistics.
        Output per-question and aggregate results to JSON.
        """
        print(f"[Layer 3 Latency] Starting latency evaluation for {len(self.questions)} questions...")
        results = []
        total_questions = len(self.questions)

        # Load dynamic evaluation data for latency metrics
        dyn_eval = self._load_dynamic_eval()
        if not dyn_eval:
            print(f"[Layer 3 Latency] ⚠️  No dynamic evaluation data found, cannot proceed")
            return results

        for i, (qid, q) in enumerate(self.questions.items(), 1):
            print(f"[Layer 3 Latency] Processing question {i}/{total_questions}: {qid}")

            # Find evaluation data for this question
            eval_data = self._find_dynamic_eval_for_qid(qid)
            if not eval_data:
                print(f"[Layer 3 Latency] ⚠️  No evaluation data found for {qid}, skipping...")
                continue

            user_question = q.get('static_question', '')
            
            # Extract latency metrics
            latency_metrics = self._extract_latency_metrics(eval_data)
            
            # Calculate latency efficiency score (1-5 scale)
            latency_efficiency = self._calculate_latency_efficiency(latency_metrics)
            
            result = {
                "question_id": qid,
                "user_request": user_question,
                "latency_metrics": latency_metrics,
                "latency_efficiency_score": latency_efficiency,
                "latency_breakdown": {
                    "response_time_seconds": latency_metrics.get('response_time_seconds', 0),
                    "response_time_ms": latency_metrics.get('response_time_ms', 0),
                    "tool_execution_time_ms": latency_metrics.get('tool_execution_time_ms', 0),
                    "processing_overhead_ms": latency_metrics.get('processing_overhead_ms', 0)
                }
            }
            
            results.append(result)
            print(f"[Layer 3 Latency] ✅ Processed {qid} - Response Time: {latency_metrics.get('response_time_seconds', 0):.3f}s, Efficiency: {latency_efficiency:.2f}/5")

        # Calculate aggregate statistics
        if results:
            response_times = [r['latency_metrics']['response_time_seconds'] for r in results if r['latency_metrics']['response_time_seconds'] > 0]
            efficiency_scores = [r['latency_efficiency_score'] for r in results]
            
            aggregate = {
                "mean_response_time": sum(response_times) / len(response_times) if response_times else 0,
                "median_response_time": sorted(response_times)[len(response_times)//2] if response_times else 0,
                "min_response_time": min(response_times) if response_times else 0,
                "max_response_time": max(response_times) if response_times else 0,
                "mean_latency_efficiency": sum(efficiency_scores) / len(efficiency_scores),
                "total_questions": len(results),
                "latency_distribution": {
                    "fast": len([t for t in response_times if t < 5]),
                    "medium": len([t for t in response_times if 5 <= t < 15]),
                    "slow": len([t for t in response_times if t >= 15])
                }
            }
            
            output = {
                "per_question": results,
                "aggregate": aggregate,
                "note": "Layer 3 Latency evaluates the response time and processing efficiency of each question."
            }
            
            # Save results
            if output_path is None:
                output_path = os.path.join(self.results_dir, "hatt_e", "layer3", "latency.json")
            
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(output, f, indent=2)
            
            print(f"[Layer 3 Latency] 💾 Results saved to {output_path}")
            print(f"[Layer 3 Latency] 📊 Aggregate - Mean Response Time: {aggregate['mean_response_time']:.3f}s, Mean Efficiency: {aggregate['mean_latency_efficiency']:.2f}/5")
        
        return results

    def _extract_latency_metrics(self, eval_data):
        """
        Extract latency-related metrics from evaluation data.
        """
        latency_metrics = {
            'response_time_seconds': 0,
            'response_time_ms': 0,
            'tool_execution_time_ms': 0,
            'processing_overhead_ms': 0
        }
        
        # Extract response time
        if 'response_time' in eval_data:
            latency_metrics['response_time_seconds'] = eval_data['response_time']
            latency_metrics['response_time_ms'] = eval_data['response_time'] * 1000
        
        # Estimate tool execution time based on number of tools used
        if 'tools_used' in eval_data:
            tool_count = len(eval_data['tools_used'])
            # Estimate 2-5 seconds per tool call
            estimated_tool_time = tool_count * 3.5  # Average 3.5 seconds per tool
            latency_metrics['tool_execution_time_ms'] = estimated_tool_time * 1000
        
        # Calculate processing overhead (response time minus tool execution)
        if latency_metrics['response_time_ms'] > 0 and latency_metrics['tool_execution_time_ms'] > 0:
            latency_metrics['processing_overhead_ms'] = max(0, latency_metrics['response_time_ms'] - latency_metrics['tool_execution_time_ms'])
        
        return latency_metrics

    def _calculate_latency_efficiency(self, latency_metrics):
        """
        Calculate latency efficiency score (1-5) based on response time and complexity.
        """
        response_time_seconds = latency_metrics.get('response_time_seconds', 0)
        
        if response_time_seconds == 0:
            return 5.0  # Perfect efficiency if no time recorded
        
        # Efficiency factors
        time_factor = max(0, 5 - (response_time_seconds * 0.2))  # Lower time = higher score
        tool_efficiency = max(0, 5 - (latency_metrics.get('tool_execution_time_ms', 0) / 1000 * 0.3))  # Fewer tool calls = higher score
        
        # Calculate weighted average
        efficiency_score = (time_factor * 0.7 + tool_efficiency * 0.3)
        
        return max(1.0, min(5.0, efficiency_score))  # Clamp between 1-5

    async def compute_layer3_turn_count(self, output_path: str = None):
        """
        Compute Turn Count metrics for Layer 3.
        Extract conversation turn counts from evaluation data and calculate efficiency metrics.
        Output per-question and aggregate results to JSON.
        """
        print(f"[Layer 3 Turn Count] Starting turn count evaluation for {len(self.questions)} questions...")
        results = []
        total_questions = len(self.questions)

        # Load dynamic evaluation data for turn count metrics
        dyn_eval = self._load_dynamic_eval()
        if not dyn_eval:
            print(f"[Layer 3 Turn Count] ⚠️  No dynamic evaluation data found, cannot proceed")
            return results

        for i, (qid, q) in enumerate(self.questions.items(), 1):
            print(f"[Layer 3 Turn Count] Processing question {i}/{total_questions}: {qid}")

            # Find evaluation data for this question
            eval_data = self._find_dynamic_eval_for_qid(qid)
            if not eval_data:
                print(f"[Layer 3 Turn Count] ⚠️  No evaluation data found for {qid}, skipping...")
                continue

            user_question = q.get('static_question', '')
            
            # Extract turn count metrics
            turn_metrics = self._extract_turn_metrics(eval_data)
            
            # Calculate turn efficiency score (1-5 scale)
            turn_efficiency = self._calculate_turn_efficiency(turn_metrics)
            
            result = {
                "question_id": qid,
                "user_request": user_question,
                "turn_metrics": turn_metrics,
                "turn_efficiency_score": turn_efficiency,
                "turn_breakdown": {
                    "total_turns": turn_metrics.get('total_turns', 0),
                    "user_turns": turn_metrics.get('user_turns', 0),
                    "agent_turns": turn_metrics.get('agent_turns', 0),
                    "tool_calls_per_turn": turn_metrics.get('tool_calls_per_turn', 0),
                    "conversation_depth": turn_metrics.get('conversation_depth', 0)
                }
            }
            
            results.append(result)
            print(f"[Layer 3 Turn Count] ✅ Processed {qid} - Total Turns: {turn_metrics.get('total_turns', 0)}, Efficiency: {turn_efficiency:.2f}/5")

        # Calculate aggregate statistics
        if results:
            total_turns = [r['turn_metrics']['total_turns'] for r in results]
            efficiency_scores = [r['turn_efficiency_score'] for r in results]
            
            aggregate = {
                "mean_total_turns": sum(total_turns) / len(total_turns),
                "median_total_turns": sorted(total_turns)[len(total_turns)//2],
                "min_total_turns": min(total_turns),
                "max_total_turns": max(total_turns),
                "mean_turn_efficiency": sum(efficiency_scores) / len(efficiency_scores),
                "total_questions": len(results),
                "turn_distribution": {
                    "simple": len([t for t in total_turns if t <= 2]),
                    "moderate": len([t for t in total_turns if 3 <= t <= 5]),
                    "complex": len([t for t in total_turns if t > 5])
                }
            }
            
            output = {
                "per_question": results,
                "aggregate": aggregate,
                "note": "Layer 3 Turn Count evaluates the conversation efficiency and complexity of each question."
            }
            
            # Save results
            if output_path is None:
                output_path = os.path.join(self.results_dir, "hatt_e", "layer3", "turn_count.json")
            
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(output, f, indent=2)
            
            print(f"[Layer 3 Turn Count] 💾 Results saved to {output_path}")
            print(f"[Layer 3 Turn Count] 📊 Aggregate - Mean Turns: {aggregate['mean_total_turns']:.2f}, Mean Efficiency: {aggregate['mean_turn_efficiency']:.2f}/5")
        
        return results

    def _extract_turn_metrics(self, eval_data):
        """
        Extract turn count-related metrics from evaluation data.
        """
        turn_metrics = {
            'total_turns': 0,
            'user_turns': 0,
            'agent_turns': 0,
            'tool_calls_per_turn': 0,
            'conversation_depth': 0
        }
        
        # Extract from dynamic conversation
        if 'dynamic_conversation' in eval_data:
            conversation = eval_data['dynamic_conversation']
            turn_metrics['total_turns'] = len(conversation)
            
            # Count user and agent turns
            user_turns = sum(1 for turn in conversation if turn.get('role') == 'user')
            agent_turns = sum(1 for turn in conversation if turn.get('role') == 'assistant')
            
            turn_metrics['user_turns'] = user_turns
            turn_metrics['agent_turns'] = agent_turns
        
        # Calculate tool calls per turn
        if 'tools_used' in eval_data and turn_metrics['total_turns'] > 0:
            tool_count = len(eval_data['tools_used'])
            turn_metrics['tool_calls_per_turn'] = tool_count / turn_metrics['total_turns']
        
        # Calculate conversation depth (complexity based on turns and tools)
        if turn_metrics['total_turns'] > 0:
            tool_complexity = turn_metrics.get('tool_calls_per_turn', 0) * 2
            turn_complexity = turn_metrics['total_turns'] * 0.5
            turn_metrics['conversation_depth'] = tool_complexity + turn_complexity
        
        return turn_metrics

    def _calculate_turn_efficiency(self, turn_metrics):
        """
        Calculate turn efficiency score (1-5) based on conversation complexity and tool usage.
        """
        total_turns = turn_metrics.get('total_turns', 0)
        tool_calls_per_turn = turn_metrics.get('tool_calls_per_turn', 0)
        conversation_depth = turn_metrics.get('conversation_depth', 0)
        
        if total_turns == 0:
            return 5.0  # Perfect efficiency if no turns recorded
        
        # Efficiency factors
        turn_factor = max(0, 5 - (total_turns * 0.5))  # Fewer turns = higher score
        tool_efficiency = max(0, 5 - (tool_calls_per_turn * 2))  # Optimal tool usage = higher score
        depth_factor = max(0, 5 - (conversation_depth * 0.3))  # Moderate complexity = higher score
        
        # Calculate weighted average
        efficiency_score = (turn_factor * 0.4 + tool_efficiency * 0.4 + depth_factor * 0.2)
        
        return max(1.0, min(5.0, efficiency_score))  # Clamp between 1-5

    def visualize_layer3_metrics(self):
        """
        Generate comprehensive visualizations for all Layer 3 metrics.
        Creates histograms, correlation plots, and combined dashboards.
        """
        print("[Layer 3 Visualization] Starting Layer 3 visualization generation...")
        
        # Create visualization directory
        vis_dir = os.path.join(self.results_dir, "hatt_e", "visualization")
        os.makedirs(vis_dir, exist_ok=True)
        
        # Load all Layer 3 metrics
        layer3_metrics = self._load_layer3_metrics()
        if not layer3_metrics:
            print("[Layer 3 Visualization] ⚠️  No Layer 3 metrics found, cannot proceed")
            return
        
        # Generate individual metric visualizations
        self._plot_tsr_distribution(layer3_metrics.get('tsr'), vis_dir)
        self._plot_rq_distribution(layer3_metrics.get('rq'), vis_dir)
        self._plot_consistency_distribution(layer3_metrics.get('consistency'), vis_dir)
        self._plot_cost_analysis(layer3_metrics.get('cost'), vis_dir)
        self._plot_latency_analysis(layer3_metrics.get('latency'), vis_dir)
        self._plot_turn_analysis(layer3_metrics.get('turns'), vis_dir)
        
        # Generate correlation plots
        self._plot_layer3_correlations(layer3_metrics, vis_dir)
        
        # Generate combined dashboard
        self._plot_layer3_dashboard(layer3_metrics, vis_dir)
        
        print(f"[Layer 3 Visualization] ✅ All visualizations saved to {vis_dir}")

    def _load_layer3_metrics(self):
        """
        Load all available Layer 3 metrics from JSON files.
        """
        layer3_dir = os.path.join(self.results_dir, "hatt_e", "layer3")
        metrics = {}
        
        # Load each metric file
        metric_files = {
            'tsr': 'task_success_rate.json',
            'rq': 'response_quality.json', 
            'consistency': 'consistency.json',
            'cost': 'system_cost.json',
            'latency': 'latency.json',
            'turns': 'turn_count.json'
        }
        
        for metric_name, filename in metric_files.items():
            filepath = os.path.join(layer3_dir, filename)
            if os.path.exists(filepath):
                try:
                    with open(filepath, 'r') as f:
                        metrics[metric_name] = json.load(f)
                    print(f"[Layer 3 Visualization] 📊 Loaded {metric_name} metrics")
                except Exception as e:
                    print(f"[Layer 3 Visualization] ⚠️  Error loading {metric_name}: {e}")
            else:
                print(f"[Layer 3 Visualization] ⚠️  {filename} not found")
        
        return metrics

    def _plot_tsr_distribution(self, tsr_data, vis_dir):
        """Plot Task Success Rate distribution."""
        if not tsr_data or 'per_question' not in tsr_data:
            return
        
        plt.figure(figsize=(10, 6))
        scores = [q['tsr'] for q in tsr_data['per_question']]
        
        plt.hist(scores, bins=20, alpha=0.7, color='green', edgecolor='black')
        plt.axvline(tsr_data['aggregate']['mean_tsr'], color='red', linestyle='--', 
                   label=f'Mean: {tsr_data["aggregate"]["mean_tsr"]:.3f}')
        plt.xlabel('Task Success Rate')
        plt.ylabel('Number of Questions')
        plt.title('Layer 3: Task Success Rate Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.savefig(os.path.join(vis_dir, 'layer3_tsr_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_rq_distribution(self, rq_data, vis_dir):
        """Plot Response Quality distribution."""
        if not rq_data or 'per_question' not in rq_data:
            return
        
        plt.figure(figsize=(10, 6))
        scores = [q['quality_score'] for q in rq_data['per_question']]
        
        plt.hist(scores, bins=20, alpha=0.7, color='blue', edgecolor='black')
        plt.axvline(rq_data['aggregate']['mean_quality_score'], color='red', linestyle='--',
                   label=f'Mean: {rq_data["aggregate"]["mean_quality_score"]:.2f}')
        plt.xlabel('Response Quality Score (1-5)')
        plt.ylabel('Number of Questions')
        plt.title('Layer 3: Response Quality Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.savefig(os.path.join(vis_dir, 'layer3_rq_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_consistency_distribution(self, consistency_data, vis_dir):
        """Plot Consistency distribution."""
        if not consistency_data or 'per_question' not in consistency_data:
            return
        
        plt.figure(figsize=(10, 6))
        scores = [q['consistency_score'] for q in consistency_data['per_question']]
        
        plt.hist(scores, bins=20, alpha=0.7, color='orange', edgecolor='black')
        plt.axvline(consistency_data['aggregate']['mean_consistency_score'], color='red', linestyle='--',
                   label=f'Mean: {consistency_data["aggregate"]["mean_consistency_score"]:.2f}')
        plt.xlabel('Consistency Score (1-5)')
        plt.ylabel('Number of Questions')
        plt.title('Layer 3: Consistency Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.savefig(os.path.join(vis_dir, 'layer3_consistency_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_cost_analysis(self, cost_data, vis_dir):
        """Plot System Cost analysis."""
        if not cost_data or 'per_question' not in cost_data:
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Cost distribution
        costs = [q['total_cost'] for q in cost_data['per_question']]
        ax1.hist(costs, bins=20, alpha=0.7, color='purple', edgecolor='black')
        ax1.axvline(cost_data['aggregate']['mean_total_cost'], color='red', linestyle='--',
                   label=f'Mean: ${cost_data["aggregate"]["mean_total_cost"]:.4f}')
        ax1.set_xlabel('Total Cost ($)')
        ax1.set_ylabel('Number of Questions')
        ax1.set_title('System Cost Distribution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Efficiency vs Cost
        efficiencies = [q['cost_efficiency_score'] for q in cost_data['per_question']]
        ax2.scatter(costs, efficiencies, alpha=0.6, color='purple')
        ax2.set_xlabel('Total Cost ($)')
        ax2.set_ylabel('Cost Efficiency Score (1-5)')
        ax2.set_title('Cost vs Efficiency')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, 'layer3_cost_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_latency_analysis(self, latency_data, vis_dir):
        """Plot Latency analysis."""
        if not latency_data or 'per_question' not in latency_data:
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Response time distribution
        times = [q['latency_metrics']['response_time_seconds'] for q in latency_data['per_question']]
        ax1.hist(times, bins=20, alpha=0.7, color='cyan', edgecolor='black')
        ax1.axvline(latency_data['aggregate']['mean_response_time'], color='red', linestyle='--',
                   label=f'Mean: {latency_data["aggregate"]["mean_response_time"]:.2f}s')
        ax1.set_xlabel('Response Time (seconds)')
        ax1.set_ylabel('Number of Questions')
        ax1.set_title('Response Time Distribution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Efficiency vs Response time
        efficiencies = [q['latency_efficiency_score'] for q in latency_data['per_question']]
        ax2.scatter(times, efficiencies, alpha=0.6, color='cyan')
        ax2.set_xlabel('Response Time (seconds)')
        ax2.set_ylabel('Latency Efficiency Score (1-5)')
        ax2.set_title('Response Time vs Efficiency')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, 'layer3_latency_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_turn_analysis(self, turns_data, vis_dir):
        """Plot Turn Count analysis."""
        if not turns_data or 'per_question' not in turns_data:
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Turn count distribution
        turns = [q['turn_metrics']['total_turns'] for q in turns_data['per_question']]
        ax1.hist(turns, bins=range(min(turns), max(turns)+2), alpha=0.7, color='brown', edgecolor='black')
        ax1.axvline(turns_data['aggregate']['mean_total_turns'], color='red', linestyle='--',
                   label=f'Mean: {turns_data["aggregate"]["mean_total_turns"]:.2f}')
        ax1.set_xlabel('Total Turns')
        ax1.set_ylabel('Number of Questions')
        ax1.set_title('Turn Count Distribution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Efficiency vs Turns
        efficiencies = [q['turn_efficiency_score'] for q in turns_data['per_question']]
        ax2.scatter(turns, efficiencies, alpha=0.6, color='brown')
        ax2.set_xlabel('Total Turns')
        ax2.set_ylabel('Turn Efficiency Score (1-5)')
        ax2.set_title('Turns vs Efficiency')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, 'layer3_turn_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_layer3_correlations(self, layer3_metrics, vis_dir):
        """Plot correlations between Layer 3 metrics."""
        if len(layer3_metrics) < 2:
            return
        
        # Extract common metrics across all available data
        correlations = {}
        metric_names = []
        
        # Find the minimum length to ensure all arrays are the same length
        min_length = float('inf')
        for metric_name, data in layer3_metrics.items():
            if data and 'per_question' in data:
                min_length = min(min_length, len(data['per_question']))
        
        if min_length == float('inf'):
            return
        
        for metric_name, data in layer3_metrics.items():
            if data and 'per_question' in data:
                metric_names.append(metric_name)
                if metric_name == 'tsr':
                    correlations[metric_name] = [q['tsr'] for q in data['per_question'][:min_length]]
                elif metric_name == 'rq':
                    correlations[metric_name] = [q['quality_score'] for q in data['per_question'][:min_length]]
                elif metric_name == 'consistency':
                    correlations[metric_name] = [q['consistency_score'] for q in data['per_question'][:min_length]]
                elif metric_name == 'cost':
                    correlations[metric_name] = [q['cost_efficiency_score'] for q in data['per_question'][:min_length]]
                elif metric_name == 'latency':
                    correlations[metric_name] = [q['latency_efficiency_score'] for q in data['per_question'][:min_length]]
                elif metric_name == 'turns':
                    correlations[metric_name] = [q['turn_efficiency_score'] for q in data['per_question'][:min_length]]
        
        if len(correlations) < 2:
            return
        
        # Create correlation matrix
        df = pd.DataFrame(correlations)
        corr_matrix = df.corr()
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                   square=True, linewidths=0.5)
        plt.title('Layer 3 Metrics Correlation Matrix')
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, 'layer3_correlations.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_layer3_dashboard(self, layer3_metrics, vis_dir):
        """Create comprehensive Layer 3 dashboard."""
        fig = plt.figure(figsize=(20, 12))
        
        # Create subplots
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
        
        # Summary statistics
        ax_summary = fig.add_subplot(gs[0, :2])
        self._plot_layer3_summary(layer3_metrics, ax_summary)
        
        # Metric distributions
        ax_dist1 = fig.add_subplot(gs[0, 2])
        ax_dist2 = fig.add_subplot(gs[0, 3])
        ax_dist3 = fig.add_subplot(gs[1, 0])
        ax_dist4 = fig.add_subplot(gs[1, 1])
        ax_dist5 = fig.add_subplot(gs[1, 2])
        ax_dist6 = fig.add_subplot(gs[1, 3])
        
        # Plot distributions
        self._plot_mini_distribution(layer3_metrics, 'tsr', 'Task Success Rate', ax_dist1, 'green')
        self._plot_mini_distribution(layer3_metrics, 'rq', 'Response Quality', ax_dist2, 'blue')
        self._plot_mini_distribution(layer3_metrics, 'consistency', 'Consistency', ax_dist3, 'orange')
        self._plot_mini_distribution(layer3_metrics, 'cost', 'Cost Efficiency', ax_dist4, 'purple')
        self._plot_mini_distribution(layer3_metrics, 'latency', 'Latency Efficiency', ax_dist5, 'cyan')
        self._plot_mini_distribution(layer3_metrics, 'turns', 'Turn Efficiency', ax_dist6, 'brown')
        
        # Performance overview
        ax_perf = fig.add_subplot(gs[2, :])
        self._plot_layer3_performance_overview(layer3_metrics, ax_perf)
        
        plt.suptitle('HATT-E Layer 3 Comprehensive Dashboard', fontsize=16, fontweight='bold')
        plt.savefig(os.path.join(vis_dir, 'layer3_comprehensive_dashboard.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_layer3_summary(self, layer3_metrics, ax):
        """Plot Layer 3 summary statistics."""
        summary_data = []
        metric_labels = []
        
        for metric_name, data in layer3_metrics.items():
            if data and 'aggregate' in data:
                agg = data['aggregate']
                if metric_name == 'tsr':
                    summary_data.append(agg.get('mean_tsr', 0))
                    metric_labels.append('TSR')
                elif metric_name == 'rq':
                    summary_data.append(agg.get('mean_quality_score', 0))
                    metric_labels.append('RQ')
                elif metric_name == 'consistency':
                    summary_data.append(agg.get('mean_consistency_score', 0))
                    metric_labels.append('Consistency')
                elif metric_name == 'cost':
                    summary_data.append(agg.get('mean_cost_efficiency', 0))
                    metric_labels.append('Cost Eff.')
                elif metric_name == 'latency':
                    summary_data.append(agg.get('mean_latency_efficiency', 0))
                    metric_labels.append('Latency Eff.')
                elif metric_name == 'turns':
                    summary_data.append(agg.get('mean_turn_efficiency', 0))
                    metric_labels.append('Turn Eff.')
        
        if summary_data:
            colors = ['green', 'blue', 'orange', 'purple', 'cyan', 'brown'][:len(summary_data)]
            bars = ax.bar(metric_labels, summary_data, color=colors, alpha=0.7)
            ax.set_ylabel('Score')
            ax.set_title('Layer 3 Metrics Summary')
            ax.set_ylim(0, 5)
            
            # Add value labels on bars
            for bar, value in zip(bars, summary_data):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                       f'{value:.2f}', ha='center', va='bottom')

    def _plot_mini_distribution(self, layer3_metrics, metric_name, title, ax, color):
        """Plot mini distribution for dashboard."""
        if metric_name not in layer3_metrics or not layer3_metrics[metric_name]:
            ax.text(0.5, 0.5, f'{title}\nNot Available', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title)
            return
        
        data = layer3_metrics[metric_name]
        if 'per_question' not in data:
            ax.text(0.5, 0.5, f'{title}\nNo Data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title)
            return
        
        # Extract scores based on metric
        if metric_name == 'tsr':
            scores = [q['tsr'] for q in data['per_question']]
        elif metric_name == 'rq':
            scores = [q['quality_score'] for q in data['per_question']]
        elif metric_name == 'consistency':
            scores = [q['consistency_score'] for q in data['per_question']]
        elif metric_name == 'cost':
            scores = [q['cost_efficiency_score'] for q in data['per_question']]
        elif metric_name == 'latency':
            scores = [q['latency_efficiency_score'] for q in data['per_question']]
        elif metric_name == 'turns':
            scores = [q['turn_efficiency_score'] for q in data['per_question']]
        else:
            scores = []
        
        if scores:
            ax.hist(scores, bins=10, alpha=0.7, color=color, edgecolor='black')
            ax.axvline(np.mean(scores), color='red', linestyle='--', alpha=0.8)
            ax.set_title(title)
            ax.set_xlabel('Score')
            ax.set_ylabel('Count')

    def _plot_layer3_performance_overview(self, layer3_metrics, ax):
        """Plot Layer 3 performance overview."""
        # Create performance indicators
        indicators = []
        labels = []
        
        for metric_name, data in layer3_metrics.items():
            if data and 'aggregate' in data:
                agg = data['aggregate']
                if metric_name == 'tsr':
                    indicators.append(agg.get('mean_tsr', 0))
                    labels.append('Task Success Rate')
                elif metric_name == 'rq':
                    indicators.append(agg.get('mean_quality_score', 0))
                    labels.append('Response Quality')
                elif metric_name == 'consistency':
                    indicators.append(agg.get('mean_consistency_score', 0))
                    labels.append('Consistency')
                elif metric_name == 'cost':
                    indicators.append(agg.get('mean_cost_efficiency', 0))
                    labels.append('Cost Efficiency')
                elif metric_name == 'latency':
                    indicators.append(agg.get('mean_latency_efficiency', 0))
                    labels.append('Latency Efficiency')
                elif metric_name == 'turns':
                    indicators.append(agg.get('mean_turn_efficiency', 0))
                    labels.append('Turn Efficiency')
        
        if indicators:
            # Create radar chart
            angles = np.linspace(0, 2 * np.pi, len(indicators), endpoint=False).tolist()
            angles += angles[:1]  # Complete the circle
            indicators += indicators[:1]
            
            ax.plot(angles, indicators, 'o-', linewidth=2, label='Layer 3 Performance')
            ax.fill(angles, indicators, alpha=0.25)
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(labels)
            ax.set_ylim(0, 5)
            ax.set_title('Layer 3 Performance Overview')
            ax.grid(True)
            
            # Add value labels
            for i, (angle, indicator) in enumerate(zip(angles[:-1], indicators[:-1])):
                ax.text(angle, indicator + 0.2, f'{indicator:.2f}', 
                       ha='center', va='center', fontweight='bold')

    async def aggregate_layer3_results(self, output_path: str = None):
        """
        Aggregate all Layer 3 metrics into a comprehensive summary.
        Combines TSR, RQ, Consistency, Cost, Latency, and Turn Count metrics.
        Output unified results to JSON.
        """
        print("[Layer 3 Aggregation] Starting Layer 3 aggregation...")
        
        # Load all Layer 3 metrics
        layer3_metrics = self._load_layer3_metrics()
        if not layer3_metrics:
            print("[Layer 3 Aggregation] ⚠️  No Layer 3 metrics found, cannot proceed")
            return
        
        # Create unified aggregation
        aggregation = {
            "layer3_summary": {},
            "per_question_combined": [],
            "performance_overview": {},
            "correlation_insights": {},
            "recommendations": []
        }
        
        # Aggregate summary statistics
        summary_stats = {}
        for metric_name, data in layer3_metrics.items():
            if data and 'aggregate' in data:
                agg = data['aggregate']
                if metric_name == 'tsr':
                    summary_stats['task_success_rate'] = {
                        'mean': agg.get('mean_tsr', 0),
                        'total_questions': agg.get('total_questions', 0),
                        'success_rate': agg.get('success_rate', 0)
                    }
                elif metric_name == 'rq':
                    summary_stats['response_quality'] = {
                        'mean': agg.get('mean_quality_score', 0),
                        'total_questions': agg.get('num_questions', 0),
                        'distribution': agg.get('quality_distribution', {})
                    }
                elif metric_name == 'consistency':
                    summary_stats['consistency'] = {
                        'mean': agg.get('mean_consistency_score', 0),
                        'total_questions': agg.get('num_questions', 0),
                        'distribution': agg.get('consistency_distribution', {})
                    }
                elif metric_name == 'cost':
                    summary_stats['system_cost'] = {
                        'mean_cost': agg.get('mean_total_cost', 0),
                        'mean_efficiency': agg.get('mean_cost_efficiency', 0),
                        'total_questions': agg.get('total_questions', 0),
                        'cost_distribution': agg.get('cost_distribution', {})
                    }
                elif metric_name == 'latency':
                    summary_stats['latency'] = {
                        'mean_response_time': agg.get('mean_response_time', 0),
                        'mean_efficiency': agg.get('mean_latency_efficiency', 0),
                        'total_questions': agg.get('total_questions', 0),
                        'latency_distribution': agg.get('latency_distribution', {})
                    }
                elif metric_name == 'turns':
                    summary_stats['turn_efficiency'] = {
                        'mean_turns': agg.get('mean_total_turns', 0),
                        'mean_efficiency': agg.get('mean_turn_efficiency', 0),
                        'total_questions': agg.get('total_questions', 0),
                        'turn_distribution': agg.get('turn_distribution', {})
                    }
        
        aggregation['layer3_summary'] = summary_stats
        
        # Create per-question combined analysis
        question_ids = set()
        for metric_name, data in layer3_metrics.items():
            if data and 'per_question' in data:
                for q in data['per_question']:
                    question_ids.add(q['question_id'])
        
        for qid in sorted(question_ids):
            combined_question = {
                "question_id": qid,
                "metrics": {}
            }
            
            # Extract metrics for this question from all available data
            for metric_name, data in layer3_metrics.items():
                if data and 'per_question' in data:
                    question_data = next((q for q in data['per_question'] if q['question_id'] == qid), None)
                    if question_data:
                        if metric_name == 'tsr':
                            combined_question['metrics']['task_success_rate'] = question_data.get('tsr', 0)
                        elif metric_name == 'rq':
                            combined_question['metrics']['response_quality'] = question_data.get('quality_score', 0)
                        elif metric_name == 'consistency':
                            combined_question['metrics']['consistency'] = question_data.get('consistency_score', 0)
                        elif metric_name == 'cost':
                            combined_question['metrics']['cost_efficiency'] = question_data.get('cost_efficiency_score', 0)
                            combined_question['metrics']['total_cost'] = question_data.get('total_cost', 0)
                        elif metric_name == 'latency':
                            combined_question['metrics']['latency_efficiency'] = question_data.get('latency_efficiency_score', 0)
                            combined_question['metrics']['response_time'] = question_data.get('latency_metrics', {}).get('response_time_seconds', 0)
                        elif metric_name == 'turns':
                            combined_question['metrics']['turn_efficiency'] = question_data.get('turn_efficiency_score', 0)
                            combined_question['metrics']['total_turns'] = question_data.get('turn_metrics', {}).get('total_turns', 0)
            
            # Calculate overall performance score
            if combined_question['metrics']:
                scores = [v for v in combined_question['metrics'].values() if isinstance(v, (int, float)) and v > 0]
                if scores:
                    combined_question['overall_performance_score'] = sum(scores) / len(scores)
                else:
                    combined_question['overall_performance_score'] = 0
            
            aggregation['per_question_combined'].append(combined_question)
        
        # Generate performance overview
        if aggregation['per_question_combined']:
            overall_scores = [q['overall_performance_score'] for q in aggregation['per_question_combined'] if 'overall_performance_score' in q]
            if overall_scores:
                aggregation['performance_overview'] = {
                    'mean_overall_score': sum(overall_scores) / len(overall_scores),
                    'min_overall_score': min(overall_scores),
                    'max_overall_score': max(overall_scores),
                    'total_questions_analyzed': len(aggregation['per_question_combined']),
                    'performance_distribution': {
                        'excellent': len([s for s in overall_scores if s >= 4.0]),
                        'good': len([s for s in overall_scores if 3.0 <= s < 4.0]),
                        'fair': len([s for s in overall_scores if 2.0 <= s < 3.0]),
                        'poor': len([s for s in overall_scores if s < 2.0])
                    }
                }
        
        # Generate correlation insights
        aggregation['correlation_insights'] = self._generate_correlation_insights(layer3_metrics)
        
        # Generate recommendations
        aggregation['recommendations'] = self._generate_layer3_recommendations(summary_stats)
        
        # Save results
        if output_path is None:
            output_path = os.path.join(self.results_dir, "hatt_e", "layer3", "layer3_aggregated.json")
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(aggregation, f, indent=2)
        
        print(f"[Layer 3 Aggregation] 💾 Aggregated results saved to {output_path}")
        print(f"[Layer 3 Aggregation] 📊 Analyzed {len(aggregation['per_question_combined'])} questions")
        
        return aggregation

    def _generate_correlation_insights(self, layer3_metrics):
        """Generate insights about correlations between Layer 3 metrics."""
        insights = {
            "strong_correlations": [],
            "weak_correlations": [],
            "performance_patterns": []
        }
        
        # Simple correlation analysis based on available metrics
        if 'tsr' in layer3_metrics and 'rq' in layer3_metrics:
            insights["strong_correlations"].append("Task Success Rate and Response Quality typically show strong correlation")
        
        if 'cost' in layer3_metrics and 'latency' in layer3_metrics:
            insights["performance_patterns"].append("Cost efficiency and latency efficiency often trade off against each other")
        
        if 'turns' in layer3_metrics and 'consistency' in layer3_metrics:
            insights["weak_correlations"].append("Turn count and consistency may show weak correlation depending on question complexity")
        
        return insights

    def _generate_layer3_recommendations(self, summary_stats):
        """Generate recommendations based on Layer 3 performance."""
        recommendations = []
        
        # Task Success Rate recommendations
        if 'task_success_rate' in summary_stats:
            tsr_mean = summary_stats['task_success_rate']['mean']
            if tsr_mean < 0.5:
                recommendations.append("Improve task success rate by enhancing agent planning and tool selection")
            elif tsr_mean < 0.8:
                recommendations.append("Optimize task success rate by refining agent responses and error handling")
        
        # Response Quality recommendations
        if 'response_quality' in summary_stats:
            rq_mean = summary_stats['response_quality']['mean']
            if rq_mean < 3.0:
                recommendations.append("Enhance response quality by improving agent knowledge and response structure")
            elif rq_mean < 4.0:
                recommendations.append("Fine-tune response quality by optimizing agent communication style")
        
        # Consistency recommendations
        if 'consistency' in summary_stats:
            consistency_mean = summary_stats['consistency']['mean']
            if consistency_mean < 3.0:
                recommendations.append("Improve consistency by standardizing agent response patterns")
        
        # Cost efficiency recommendations
        if 'system_cost' in summary_stats:
            cost_efficiency = summary_stats['system_cost']['mean_efficiency']
            if cost_efficiency < 3.0:
                recommendations.append("Optimize system cost by reducing unnecessary API calls and token usage")
        
        # Latency recommendations
        if 'latency' in summary_stats:
            latency_efficiency = summary_stats['latency']['mean_efficiency']
            if latency_efficiency < 3.0:
                recommendations.append("Improve latency efficiency by optimizing response time and processing")
        
        # Turn efficiency recommendations
        if 'turn_efficiency' in summary_stats:
            turn_efficiency = summary_stats['turn_efficiency']['mean_efficiency']
            if turn_efficiency < 3.0:
                recommendations.append("Optimize conversation efficiency by reducing unnecessary turns")
        
        return recommendations

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python hatt_e_metrics.py <results_dir>")
        sys.exit(1)
    results_dir = sys.argv[1]
    
    # Run complete HATT-E pipeline (Layer 1 + Layer 2 + Layer 3)
    evaluator = HATTEvaluator(results_dir)
    
    # Layer 1 metrics and visualization
    print("[HATT-E] Running Layer 1 metrics...")
    dqs_llm_output_path = os.path.join(evaluator.output_dir, "hatt_e_dqs_llm.json")
    import asyncio
    asyncio.run(evaluator.evaluate_dqs_llm_for_all(dqs_llm_output_path))
    evaluator.print_dqs_prompts_for_all()
    da_output_path = os.path.join(evaluator.output_dir, "hatt_e_delegation_accuracy.json")
    evaluator.compute_delegation_accuracy_report(da_output_path)
    failure_modes_output_path = os.path.join(evaluator.output_dir, "hatt_e_failure_modes.json")
    evaluator.tag_failure_modes(dqs_llm_output_path, failure_modes_output_path)
    aggregated_output_path = os.path.join(evaluator.output_dir, "hatt_e_layer1_aggregated.json")
    evaluator.aggregate_layer1_results(dqs_llm_output_path, da_output_path, failure_modes_output_path, aggregated_output_path)
    evaluator.visualize_layer1_metrics()
    print(f"[HATT-E] Layer 1 metrics, aggregated results, and visualizations written to {evaluator.output_dir} and visualization folder.")
    
    # Layer 2 metrics
    print("[HATT-E] Running Layer 2 metrics...")
    l2_dir = os.path.join(results_dir, "hatt_e", "layer2")
    os.makedirs(l2_dir, exist_ok=True)
    planem_path = os.path.join(l2_dir, "plan_em_results.json")
    actem_path = os.path.join(l2_dir, "act_em_results.json")
    halluc_path = os.path.join(l2_dir, "hallucination_results.json")
    tsr_path = os.path.join(l2_dir, "tsr_results.json")
    evaluator.compute_plan_em_for_all(output_path=planem_path, use_llm=True)
    print(f"[Layer 2] Plan.EM results saved to {planem_path}")
    evaluator.compute_act_em_for_all(output_path=actem_path)
    print(f"[Layer 2] Act.EM results saved to {actem_path}")
    evaluator.compute_hallucination_rate_for_all(output_path=halluc_path)
    print(f"[Layer 2] Hallucination Rate results saved to {halluc_path}")
    evaluator.compute_tsr_for_all(output_path=tsr_path)
    print(f"[Layer 2] TSR results saved to {tsr_path}")
    evaluator.visualize_layer2_metrics()
    print("[HATT-E] Layer 2 visualizations written to hatt_e/visualization/")
    
    # Layer 3 metrics and visualization
    print("[HATT-E] Running Layer 3 metrics...")
    l3_dir = os.path.join(results_dir, "hatt_e", "layer3")
    os.makedirs(l3_dir, exist_ok=True)
    tsr3_path = os.path.join(l3_dir, "task_success_rate.json")
    rq_path = os.path.join(l3_dir, "response_quality.json")
    consistency_path = os.path.join(l3_dir, "consistency.json")
    cost_path = os.path.join(l3_dir, "system_cost.json")
    latency_path = os.path.join(l3_dir, "latency.json")
    turns_path = os.path.join(l3_dir, "turn_count.json")
    
    # Run all Layer 3 metrics
    print("[Layer 3] Computing Task Success Rate (LLM-judged)...")
    asyncio.run(evaluator.compute_layer3_tsr_llm(output_path=tsr3_path))
    print(f"[Layer 3] Task Success Rate results saved to {tsr3_path}")
    
    print("[Layer 3] Computing Response Quality...")
    asyncio.run(evaluator.compute_layer3_response_quality(output_path=rq_path))
    print(f"[Layer 3] Response Quality results saved to {rq_path}")
    
    print("[Layer 3] Computing Consistency...")
    asyncio.run(evaluator.compute_layer3_consistency(output_path=consistency_path))
    print(f"[Layer 3] Consistency results saved to {consistency_path}")
    
    print("[Layer 3] Computing System Cost...")
    asyncio.run(evaluator.compute_layer3_system_cost(output_path=cost_path))
    print(f"[Layer 3] System Cost results saved to {cost_path}")
    
    print("[Layer 3] Computing Latency...")
    asyncio.run(evaluator.compute_layer3_latency(output_path=latency_path))
    print(f"[Layer 3] Latency results saved to {latency_path}")
    
    print("[Layer 3] Computing Turn Count...")
    asyncio.run(evaluator.compute_layer3_turn_count(output_path=turns_path))
    print(f"[Layer 3] Turn Count results saved to {turns_path}")
    
    # Generate Layer 3 visualizations
    print("[Layer 3] Generating visualizations...")
    evaluator.visualize_layer3_metrics()
    print(f"[Layer 3] Visualizations saved to {results_dir}/hatt_e/visualization/")
    
    # Generate Layer 3 aggregation
    print("[Layer 3] Generating aggregation...")
    asyncio.run(evaluator.aggregate_layer3_results())
    print(f"[Layer 3] Aggregation saved to {results_dir}/hatt_e/layer3/layer3_aggregated.json")
    
    print("[HATT-E] Complete HATT-E evaluation pipeline finished!")
    print(f"[HATT-E] All results saved to {results_dir}/hatt_e/")
    print(f"[HATT-E] All visualizations saved to {results_dir}/hatt_e/visualization/") 