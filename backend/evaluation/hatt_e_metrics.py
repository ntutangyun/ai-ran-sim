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

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python hatt_e_metrics.py <results_dir>")
        sys.exit(1)
    results_dir = sys.argv[1]
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