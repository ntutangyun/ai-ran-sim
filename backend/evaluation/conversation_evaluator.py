"""
Dynamic Conversation Evaluator using OpenAI Agents SDK
"""

import asyncio
import json
import time
import logging
import os
import sys
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime

# Add backend to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Set up logging to both stdout and a debug file
log_formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Stream handler for stdout
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(log_formatter)
logger.addHandler(stream_handler)

# File handler for debug log (will be set to correct path in __init__)
file_handler = None

def set_debug_log_file(log_path):
    global file_handler
    if file_handler:
        logger.removeHandler(file_handler)
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(log_formatter)
    logger.addHandler(file_handler)

# Set up OpenAI API key before importing agents
from agents import set_default_openai_key, set_tracing_disabled

# Check if API key is set
api_key = os.environ.get("OPENAI_API_KEY", "")
if not api_key:
    raise ValueError("OPENAI_API_KEY environment variable is not set. Please set it with: export OPENAI_API_KEY='your-api-key-here'")

set_default_openai_key(api_key)
set_tracing_disabled(True)

from agents import Agent, Runner
from agents.extensions.handoff_prompt import RECOMMENDED_PROMPT_PREFIX
from intelligence_layer.engineer_chat_agent import engineer_chat_agent
from intelligence_layer.network_knowledge_agent import (
    non_reasoning_network_knowledge_agent,
    reasoning_network_knowledge_agent
)
from knowledge_layer import KnowledgeRouter
from network_layer.simulation_engine import SimulationEngine
from settings import OPENAI_NON_REASONING_MODEL_NAME, SIM_STEP_TIME_DEFAULT
from utils import WebSocketSingleton, setup_logging

logger = logging.getLogger(__name__)


@dataclass
class ConversationResult:
    question_id: str
    difficulty: str
    category: str
    static_question: str
    dynamic_conversation: List[Dict[str, str]]
    agent_response: str
    tools_used: List[str]
    tool_outputs: Dict[str, str]
    response_time: float
    evaluation_score: float
    evaluation_reasoning: str
    timestamp: datetime
    conversation_log: str


class DynamicConversationEvaluator:
    """
    Evaluates the Network Engineer Chat Agent using dynamic conversations
    """
    
    def __init__(self, conversation_data_file: str = "conversation_data.json", debug_log_dir: Optional[str] = None):
        self.conversation_data_file = conversation_data_file
        self.simulation_engine = None
        self.knowledge_router = None
        self.conversation_ai = self._create_conversation_ai()
        self.evaluation_ai = self._create_evaluation_ai()
        self.results = []
        self.log_buffer = []
        # Set debug log file in results dir if provided
        if debug_log_dir:
            os.makedirs(debug_log_dir, exist_ok=True)
            set_debug_log_file(os.path.join(debug_log_dir, "debug_agent_events.log"))
    
    async def _delay_between_calls(self, delay_seconds: float = 0.5):
        """Add delay between LLM calls to prevent rate limiting"""
        await asyncio.sleep(delay_seconds)
    
    def _create_conversation_ai(self) -> Agent:
        """Create AI agent for dynamic conversation generation"""
        return Agent(
            name="Conversation Simulator",
            instructions=f"""{RECOMMENDED_PROMPT_PREFIX}
You are a network engineer who asks questions about the telecom network. 
Given a static question, you should:
1. Ask follow-up questions naturally as a real engineer would
2. Show interest in the response and ask for more details
3. Ask clarifying questions if the response is unclear
4. Continue the conversation for 14 turns maximum if asked for clarification, but don't deviate from the original question
5. Keep the conversation focused on the original topic

Example conversation flow:
- User: "What is the status of UE_001?"
- You: "Can you also tell me about its connection quality and which cell it's connected to?"
- User: "What about its performance metrics?"
- You: "Thanks, that's helpful. Is there anything unusual about its behavior?"

Be natural and conversational, not robotic.
""",
            model=OPENAI_NON_REASONING_MODEL_NAME,
        )
    
    def _create_evaluation_ai(self) -> Agent:
        """Create AI agent for evaluating responses"""
        return Agent(
            name="Response Evaluator",
            instructions=f"""{RECOMMENDED_PROMPT_PREFIX}
You are an expert evaluator of network engineer responses. Your job is to evaluate if a response is accurate and helpful.

Evaluation criteria:
1. **Tool Usage**: Did the agent use appropriate tools when expected? (0-1 score)
2. **Accuracy**: Is the response accurate based on the tool outputs? (0-1 score)
3. **Completeness**: Does the response address the question fully? (0-1 score)
4. **Clarity**: Is the response clear and understandable? (0-1 score)
5. **Helpfulness**: Is the response actually helpful to a network engineer? (0-1 score)

For each response, provide your evaluation in this EXACT format:

SCORE: [provide a specific score from 0.0 to 1.0, not just 0.5]
REASONING: [detailed explanation of why you gave this score]
FEEDBACK: [specific feedback on what was good/bad]

If tools were used, compare the response with the actual tool outputs.
If no tools were used, evaluate based on general accuracy and helpfulness.

IMPORTANT: 
- Provide specific scores, not generic 0.5 scores
- Be critical and detailed in your evaluation
- Use the exact format above with SCORE:, REASONING:, and FEEDBACK: labels
""",
            model=OPENAI_NON_REASONING_MODEL_NAME,
        )
    
    def _initialize_simulation(self):
        """Initialize simulation like the frontend does"""
        logger.info("Initializing simulation...")
        
        # Setup logging
        setup_logging()
        
        # Initialize simulation engine
        self.simulation_engine = SimulationEngine()
        self.simulation_engine.reset_network()
        self.simulation_engine.network_setup()
        
        # Initialize knowledge router
        self.knowledge_router = KnowledgeRouter()
        self.knowledge_router.import_routes(self.simulation_engine)
        
        # Start simulation in background (non-blocking)
        logger.info("Starting simulation in background...")
        asyncio.create_task(self._run_simulation_background())
        
        # Wait a bit for simulation to initialize
        time.sleep(5)
        logger.info("Simulation initialized successfully")
    
    async def _run_simulation_background(self):
        """Run simulation in background without blocking"""
        try:
            # Run simulation for a few steps to initializes
            for i in range(3):  # Run 3 simulation steps for initialization
                if self.simulation_engine:
                    self.simulation_engine.step(SIM_STEP_TIME_DEFAULT)
                await asyncio.sleep(0.1)  # Small delay between steps
            logger.info("Background simulation initialized with 3 steps")
        except Exception as e:
            logger.error(f"Error in background simulation: {e}")
    
    def _log_conversation(self, question_id: str, conversation_data: Dict[str, Any]):
        """Log the entire conversation for debugging"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "question_id": question_id,
            "static_question": conversation_data.get("static_question", ""),
            "dynamic_conversation": conversation_data.get("dynamic_conversation", []),
            "agent_response": conversation_data.get("agent_response", ""),
            "tools_used": conversation_data.get("tools_used", []),
            "tool_outputs": conversation_data.get("tool_outputs", {}),
            "response_time": conversation_data.get("response_time", 0),
            "evaluation_score": conversation_data.get("evaluation_score", 0),
            "evaluation_reasoning": conversation_data.get("evaluation_reasoning", ""),
            "max_turns_exceeded": conversation_data.get("max_turns_exceeded", False)
        }
        
        self.log_buffer.append(log_entry)
        
        # Create detailed log text
        log_text = f"""
=== CONVERSATION LOG FOR {question_id} ===
Timestamp: {log_entry['timestamp']}
Static Question: {log_entry['static_question']}

DYNAMIC CONVERSATION:
"""
        
        for i, turn in enumerate(log_entry['dynamic_conversation']):
            role = turn.get('role', 'user')
            content = turn.get('content', '')
            if not content.strip():
                content = '[NO RESPONSE]'
            log_text += f"Turn {i+1} [{role}]: {content}\n"
        # Add max turns exceeded warning if applicable
        if log_entry.get('max_turns_exceeded', False):
            log_text += f"\n⚠️  WARNING: Max turns (14) exceeded - conversation was terminated\n"
        
        log_text += f"""
AGENT RESPONSE:
{log_entry['agent_response']}

TOOLS USED: {log_entry['tools_used']}

TOOL OUTPUTS:
"""
        
        for tool, output in log_entry['tool_outputs'].items():
            log_text += f"{tool}: {output}\n"
        
        log_text += f"""
RESPONSE TIME: {log_entry['response_time']:.2f}s
EVALUATION SCORE: {log_entry['evaluation_score']:.2f}
EVALUATION REASONING:
{log_entry['evaluation_reasoning']}

=== END LOG ===
"""
        
        return log_text
    
    def load_conversation_data(self) -> Dict[str, Any]:
        """Load conversation data from JSON file"""
        try:
            with open(self.conversation_data_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.error(f"Conversation data file not found: {self.conversation_data_file}")
            return {"questions": [], "metadata": {}}
    
    async def evaluate_response_with_tools(self, question: Dict[str, Any], agent_response: str, tools_used: List[str], tool_outputs: Dict[str, str]) -> tuple[float, str]:
        """Evaluate response by comparing with tool outputs"""
        try:
            # Add delay before evaluation
            await self._delay_between_calls(0.5)
            
            evaluation_prompt = f"""
Question: {question['static_question']}
Expected tools: {question['expected_tools']}
Actual tools used: {tools_used}
Agent response: {agent_response}
Tool outputs: {tool_outputs}

Evaluation criteria:
1. Did the agent use the expected tools? (Score: 0-1)
2. Is the response accurate based on tool outputs? (Score: 0-1)
3. Is the response complete and helpful? (Score: 0-1)

Provide your evaluation in this EXACT format:

SCORE: [provide a specific score from 0.0 to 1.0, not just 0.5]
REASONING: [detailed explanation of why you gave this score]
FEEDBACK: [specific feedback on accuracy, completeness, and tool usage]

IMPORTANT: 
- Provide specific scores, not generic 0.5 scores
- Be critical and detailed in your evaluation
- Use the exact format above with SCORE:, REASONING:, and FEEDBACK: labels
"""
            
            evaluation_streamer = Runner.run_streamed(self.evaluation_ai, evaluation_prompt)
            
            evaluation_text = ""
            async for event in evaluation_streamer.stream_events():
                if event.type == "raw_response_event":
                    # For OpenAI Agents SDK, the delta is the new text chunk
                    delta = getattr(event.data, "delta", None)
                    if isinstance(delta, str):
                        evaluation_text += delta
                    elif isinstance(event.data, str):
                        evaluation_text += event.data
                elif event.type == "message_output_item":
                    if isinstance(event.data, str):
                        evaluation_text += event.data
                elif event.type == "run_completed":
                    break
            
            # Extract score from evaluation (improved parsing)
            lines = evaluation_text.split('\n')
            score = 0.5  # Default score
            
            for line in lines:
                line_lower = line.lower()
                if "score:" in line_lower:
                    try:
                        # Extract number from line
                        import re
                        numbers = re.findall(r'\d+\.?\d*', line)
                        if numbers:
                            score = float(numbers[0])
                            if score > 1.0:
                                score = score / 100  # Handle percentage format
                            break
                    except:
                        continue
            
            return score, evaluation_text
            
        except Exception as e:
            logger.error(f"Error evaluating response: {e}")
            return 0.5, f"Evaluation error: {str(e)}"
    
    async def evaluate_response_without_tools(self, question: Dict[str, Any], agent_response: str) -> tuple[float, str]:
        """Evaluate response when no tools are used"""
        try:
            # Add delay before evaluation
            await self._delay_between_calls(0.5)
            
            evaluation_prompt = f"""
Question: {question['static_question']}
Agent response: {agent_response}
Self-evaluation prompt: {question['evaluation_criteria']['self_evaluation_prompt']}

Evaluate the response based on:
1. Accuracy of the information provided (0-1 score)
2. Completeness of the answer (0-1 score)
3. Helpfulness to a network engineer (0-1 score)
4. Clarity and understandability (0-1 score)

Provide your evaluation in this EXACT format:

SCORE: [provide a specific score from 0.0 to 1.0, not just 0.5]
REASONING: [detailed explanation of why you gave this score]
FEEDBACK: [specific feedback on what was good/bad]

IMPORTANT: 
- Provide specific scores, not generic 0.5 scores
- Be critical and detailed in your evaluation
- Use the exact format above with SCORE:, REASONING:, and FEEDBACK: labels
"""
            
            evaluation_streamer = Runner.run_streamed(self.evaluation_ai, evaluation_prompt)
            
            evaluation_text = ""
            async for event in evaluation_streamer.stream_events():
                if event.type == "raw_response_event":
                    # For OpenAI Agents SDK, the delta is the new text chunk
                    delta = getattr(event.data, "delta", None)
                    if isinstance(delta, str):
                        evaluation_text += delta
                    elif isinstance(event.data, str):
                        evaluation_text += event.data
                elif event.type == "message_output_item":
                    if isinstance(event.data, str):
                        evaluation_text += event.data
                elif event.type == "run_completed":
                    break
            
            # Extract score from evaluation (improved parsing)
            lines = evaluation_text.split('\n')
            score = 0.5  # Default score
            
            for line in lines:
                line_lower = line.lower()
                if "score:" in line_lower:
                    try:
                        # Extract number from line
                        import re
                        numbers = re.findall(r'\d+\.?\d*', line)
                        if numbers:
                            score = float(numbers[0])
                            if score > 1.0:
                                score = score / 100  # Handle percentage format
                            break
                    except:
                        continue
            
            return score, evaluation_text
            
        except Exception as e:
            logger.error(f"Error evaluating response without tools: {e}")
            return 0.5, f"Evaluation error: {str(e)}"
    
    async def run_conversation_evaluation(self, question: Dict[str, Any]) -> ConversationResult:
        """Run evaluation for a single question with proper multi-turn conversation"""
        logger.info(f"Evaluating question: {question['id']} - {question['static_question']}")
        
        # Initialize conversation tracking
        conversation_turns = []
        current_question = question['static_question']
        max_turns = 14 
        turn_count = 0
        max_turns_exceeded = False
        
        # Run multi-turn conversation
        start_time = time.time()
        
        try:
            while turn_count < max_turns:
                logger.info(f"Conversation turn {turn_count + 1}: {current_question}")
                
                # Add user question to conversation (only if not a duplicate of previous user turn)
                if not (len(conversation_turns) > 0 and conversation_turns[-1]["role"] == "user" and conversation_turns[-1]["content"] == current_question):
                    conversation_turns.append({
                        "role": "user",
                        "content": current_question
                    })
                
                # Add delay before LLM call
                await self._delay_between_calls(0.5)
                
                # Send full conversation history to agent
                conversation_streamer = Runner.run_streamed(engineer_chat_agent, conversation_turns)
                
                agent_response = ""
                tools_used = []
                tool_outputs = {}
                
                # Capture agent response
                async for event in conversation_streamer.stream_events():
                    if event.type == "raw_response_event":
                        delta = getattr(event.data, "delta", None)
                        if isinstance(delta, str):
                            agent_response += delta
                        elif isinstance(event.data, str):
                            agent_response += event.data
                    elif event.type == "message_output_item":
                        if isinstance(event.data, str):
                            agent_response += event.data
                    elif event.type == "run_item_stream_event":
                        if event.item.type == "tool_call_item":
                            tool_name = getattr(event.item.raw_item, 'name', 'unknown_tool')
                            tools_used.append(tool_name)
                        elif event.item.type == "tool_call_output_item":
                            tool_output = event.item.raw_item["output"]
                            if tools_used:
                                tool_outputs[tools_used[-1]] = tool_output
                    elif event.type == "run_completed":
                        break
                
                # Add agent response to conversation
                conversation_turns.append({
                    "role": "assistant", 
                    "content": agent_response
                })
                
                logger.info(f"Agent response: {agent_response}")
                
                # Increment turn count for this response
                turn_count += 1
                
                # Check if we've reached max turns
                if turn_count >= max_turns:
                    max_turns_exceeded = True
                    logger.warning(f"Max turns ({max_turns}) exceeded for question {question['id']}")
                    break
                
                # Let the LLM decide if the conversation should continue
                # Generate a follow-up response to continue the conversation naturally
                await self._delay_between_calls(0.5)
                
                # Create a prompt for the conversation AI to decide on next turn
                conversation_prompt = f"""
You are evaluating a network engineer agent's response to this test question: "{question['static_question']}"

The agent just responded with: "{agent_response}"

Your task is to determine if the conversation should continue or end:

**STOP the conversation if:**
- The agent provided a complete, relevant answer to the test question
- The agent's response directly addresses what was asked
- The agent is just offering generic help like "let me know if you need more details" or "if you have other questions"

**CONTINUE the conversation only if:**
- The agent is asking for genuine clarification about the test question itself
- The agent needs more specific information to answer the test question
- The response is unclear or incomplete for the test question

**IMPORTANT:** The test question is "{question['static_question']}". Only continue if the agent is asking about this specific question, not offering general help.

If the agent's response is complete and satisfactory for the test question, respond with "CONVERSATION_COMPLETE".
If you need to ask a follow-up question, make it specific to the test question.
"""
                
                conversation_streamer = Runner.run_streamed(self.conversation_ai, conversation_prompt)
                
                follow_up_response = ""
                async for event in conversation_streamer.stream_events():
                    if event.type == "raw_response_event":
                        delta = getattr(event.data, "delta", None)
                        if isinstance(delta, str):
                            follow_up_response += delta
                        elif isinstance(event.data, str):
                            follow_up_response += event.data
                    elif event.type == "message_output_item":
                        if isinstance(event.data, str):
                            follow_up_response += event.data
                    elif event.type == "run_completed":
                        break
                
                # Check if the conversation should end
                if "CONVERSATION_COMPLETE" in follow_up_response.upper():
                    logger.info("LLM determined conversation is complete")
                    break
                
                # Add the follow-up response to conversation (only if not a duplicate of previous user turn)
                if not (len(conversation_turns) > 0 and conversation_turns[-1]["role"] == "user" and conversation_turns[-1]["content"] == follow_up_response.strip()):
                    conversation_turns.append({
                        "role": "user",
                        "content": follow_up_response.strip()
                    })
                
                current_question = follow_up_response.strip()
                continue
            
            response_time = time.time() - start_time
            
            # Get the final agent response (last assistant message)
            final_agent_response = ""
            for turn in reversed(conversation_turns):
                if turn["role"] == "assistant":
                    final_agent_response = turn["content"]
                    break
            
            logger.info(f"Final agent response: '{final_agent_response}'")
            logger.info(f"Tools used: {tools_used}")
            logger.info(f"Tool outputs: {tool_outputs}")
            logger.info(f"Total conversation turns: {len(conversation_turns)}")
            
            # Add delay before evaluation
            await self._delay_between_calls(0.5)
            
            # Evaluate response
            if tools_used and question['evaluation_criteria']['should_use_tools']:
                evaluation_score, evaluation_reasoning = await self.evaluate_response_with_tools(
                    question, final_agent_response, tools_used, tool_outputs
                )
            else:
                evaluation_score, evaluation_reasoning = await self.evaluate_response_without_tools(
                    question, final_agent_response
                )
            
            # Adjust score if max turns were exceeded
            if max_turns_exceeded:
                evaluation_score = max(0.0, evaluation_score - 0.2)  # Penalize for not completing
                evaluation_reasoning += f"\n\nNOTE: Max turns ({max_turns}) exceeded. Score reduced by 0.2 points."
                logger.warning(f"Max turns exceeded - adjusted score from {evaluation_score + 0.2:.2f} to {evaluation_score:.2f}")
            
            # Create conversation log
            conversation_data = {
                "static_question": question['static_question'],
                "dynamic_conversation": conversation_turns,
                "agent_response": final_agent_response,
                "tools_used": tools_used,
                "tool_outputs": tool_outputs,
                "response_time": response_time,
                "evaluation_score": evaluation_score,
                "evaluation_reasoning": evaluation_reasoning,
                "max_turns_exceeded": max_turns_exceeded
            }
            
            conversation_log = self._log_conversation(question['id'], conversation_data)
            
            result = ConversationResult(
                question_id=question['id'],
                difficulty=question['difficulty'],
                category=question['category'],
                static_question=question['static_question'],
                dynamic_conversation=conversation_turns,
                agent_response=final_agent_response,
                tools_used=tools_used,
                tool_outputs=tool_outputs,
                response_time=response_time,
                evaluation_score=evaluation_score,
                evaluation_reasoning=evaluation_reasoning,
                timestamp=datetime.now(),
                conversation_log=conversation_log
            )
            
            logger.info(f"Evaluation completed for {question['id']}: Score {evaluation_score:.2f}")
            return result
            
        except Exception as e:
            logger.error(f"Error in conversation evaluation: {e}")
            
            # Check if this is a max turns exceeded error
            if "Max turns" in str(e):
                # Handle max turns exceeded gracefully
                logger.warning(f"Max turns exceeded - logging conversation and providing score")
                
                # Get the final agent response from conversation turns
                final_agent_response = ""
                for turn in reversed(conversation_turns):
                    if turn["role"] == "assistant":
                        final_agent_response = turn["content"]
                        break
                
                # Provide a default evaluation score for max turns exceeded
                evaluation_score = 0.3  # Low score for incomplete conversation
                evaluation_reasoning = f"Conversation was terminated due to max turns (14) being exceeded. Score reduced due to incomplete conversation.\n\nNOTE: Max turns (14) exceeded. Score reduced due to incomplete conversation."
                
                # Create conversation log
                conversation_data = {
                    "static_question": question['static_question'],
                    "dynamic_conversation": conversation_turns,
                    "agent_response": final_agent_response,
                    "tools_used": [],
                    "tool_outputs": {},
                    "response_time": time.time() - start_time,
                    "evaluation_score": evaluation_score,
                    "evaluation_reasoning": evaluation_reasoning,
                    "max_turns_exceeded": True
                }
                
                conversation_log = self._log_conversation(question['id'], conversation_data)
                
                return ConversationResult(
                    question_id=question['id'],
                    difficulty=question['difficulty'],
                    category=question['category'],
                    static_question=question['static_question'],
                    dynamic_conversation=conversation_turns,
                    agent_response=final_agent_response,
                    tools_used=[],
                    tool_outputs={},
                    response_time=time.time() - start_time,
                    evaluation_score=evaluation_score,
                    evaluation_reasoning=evaluation_reasoning,
                    timestamp=datetime.now(),
                    conversation_log=conversation_log
                )
            else:
                # Handle other errors as before
                return ConversationResult(
                    question_id=question['id'],
                    difficulty=question['difficulty'],
                    category=question['category'],
                    static_question=question['static_question'],
                    dynamic_conversation=conversation_turns,
                    agent_response=f"Error: {str(e)}",
                    tools_used=[],
                    tool_outputs={},
                    response_time=time.time() - start_time,
                    evaluation_score=0.0,
                    evaluation_reasoning=f"Evaluation failed: {str(e)}",
                    timestamp=datetime.now(),
                    conversation_log=f"Error in evaluation: {str(e)}"
                )
    
    async def run_comprehensive_evaluation(self, output_dir: Optional[str] = None) -> Dict[str, Any]:
        """Run comprehensive evaluation across all questions"""
        logger.info("Starting comprehensive dynamic conversation evaluation")
        
        # Initialize simulation
        self._initialize_simulation()
        
        # Load conversation data
        conversation_data = self.load_conversation_data()
        questions = conversation_data.get("questions", [])
        
        if not questions:
            logger.error("No questions found in conversation data")
            return {"error": "No questions found"}
        
        # Create timestamped output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if output_dir is None:
            output_dir = f"evaluation_results_{timestamp}"
        else:
            output_dir = str(output_dir)
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Run evaluation for each question
        results = []
        for i, question in enumerate(questions):
            logger.info(f"Processing question {i+1}/{len(questions)}: {question['id']}")
            result = await self.run_conversation_evaluation(question)
            results.append(result)
            self.results.append(result)
            
            # Save individual conversation log
            indiv_log_dir = os.path.join(output_dir, 'individual_conversation_logs')
            os.makedirs(indiv_log_dir, exist_ok=True)
            log_file = os.path.join(indiv_log_dir, f"conversation_{question['id']}.log")
            with open(log_file, 'w', encoding='utf-8') as f:
                f.write(result.conversation_log)
        
        # Generate comprehensive report
        report = self._generate_evaluation_report(results, conversation_data)
        
        # Save results
        self._save_evaluation_results(report, output_dir)
        
        logger.info(f"Comprehensive evaluation completed. Results saved to {output_dir}")
        return report
    
    def _generate_evaluation_report(self, results: List[ConversationResult], conversation_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive evaluation report"""
        
        # Calculate aggregate scores
        total_questions = len(results)
        avg_score = sum(r.evaluation_score for r in results) / total_questions if results else 0.0
        avg_response_time = sum(r.response_time for r in results) / total_questions if results else 0.0
        
        # Scores by difficulty
        difficulty_scores = {}
        for difficulty in ["easy", "medium", "hard"]:
            difficulty_results = [r for r in results if r.difficulty == difficulty]
            if difficulty_results:
                difficulty_scores[difficulty] = sum(r.evaluation_score for r in difficulty_results) / len(difficulty_results)
            else:
                difficulty_scores[difficulty] = 0.0
        
        # Tool usage analysis
        tool_usage_stats = {}
        for result in results:
            for tool in result.tools_used:
                if tool not in tool_usage_stats:
                    tool_usage_stats[tool] = 0
                tool_usage_stats[tool] += 1
        
        # Category analysis
        category_scores = {}
        for result in results:
            if result.category not in category_scores:
                category_scores[result.category] = []
            category_scores[result.category].append(result.evaluation_score)
        
        for category in category_scores:
            category_scores[category] = sum(category_scores[category]) / len(category_scores[category])
        
        return {
            "evaluation_timestamp": datetime.now().isoformat(),
            "total_questions": total_questions,
            "average_score": avg_score,
            "average_response_time": avg_response_time,
            "difficulty_scores": difficulty_scores,
            "category_scores": category_scores,
            "tool_usage_stats": tool_usage_stats,
            "detailed_results": [
                {
                    "question_id": r.question_id,
                    "difficulty": r.difficulty,
                    "category": r.category,
                    "static_question": r.static_question,
                    "dynamic_conversation": r.dynamic_conversation,
                    "agent_response": r.agent_response,
                    "tools_used": r.tools_used,
                    "tool_outputs": r.tool_outputs,
                    "response_time": r.response_time,
                    "evaluation_score": r.evaluation_score,
                    "evaluation_reasoning": r.evaluation_reasoning,
                    "timestamp": r.timestamp.isoformat()
                }
                for r in results
            ],
            "metadata": conversation_data.get("metadata", {})
        }
    
    def _save_evaluation_results(self, report: Dict[str, Any], output_dir: str):
        """Save evaluation results to files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save comprehensive report
        report_file = os.path.join(output_dir, f"dynamic_conversation_evaluation_{timestamp}.json")
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, default=str, ensure_ascii=False)
        
        # Save summary report
        summary = {
            "evaluation_timestamp": report["evaluation_timestamp"],
            "total_questions": report["total_questions"],
            "average_score": report["average_score"],
            "average_response_time": report["average_response_time"],
            "difficulty_scores": report["difficulty_scores"],
            "category_scores": report["category_scores"],
            "tool_usage_stats": report["tool_usage_stats"]
        }
        
        summary_file = os.path.join(output_dir, f"evaluation_summary_{timestamp}.json")
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, default=str, ensure_ascii=False)
        
        # Save all conversation logs
        all_logs_file = os.path.join(output_dir, f"all_conversation_logs_{timestamp}.txt")
        with open(all_logs_file, 'w', encoding='utf-8') as f:
            for result in self.results:
                f.write(result.conversation_log)
                f.write("\n" + "="*80 + "\n\n")
        
        logger.info(f"Evaluation results saved to {output_dir}")


async def main():
    """Main function for running dynamic conversation evaluation"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run dynamic conversation evaluation")
    parser.add_argument("--output-dir", default=None, help="Output directory for results")
    parser.add_argument("--conversation-data", default="conversation_data.json", help="Conversation data file")
    parser.add_argument("--test-single", default=None, help="Test a single question by ID (e.g., q_001)")
    
    args = parser.parse_args()
    
    # Create output directory for debug logs
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output_dir is None:
        output_dir = f"evaluation_results_{timestamp}"
    else:
        output_dir = str(args.output_dir)
    
    evaluator = DynamicConversationEvaluator(args.conversation_data, output_dir)
    
    if args.test_single:
        # Test single question
        logger.info(f"Testing single question: {args.test_single}")
        conversation_data = evaluator.load_conversation_data()
        questions = conversation_data.get("questions", [])
        
        # Find the question by ID
        question = None
        for q in questions:
            if q['id'] == args.test_single:
                question = q
                break
        
        if question is None:
            logger.error(f"Question {args.test_single} not found")
            return
        
        # Initialize simulation
        evaluator._initialize_simulation()
        
        # Run single evaluation
        result = await evaluator.run_conversation_evaluation(question)
        
        # Save result
        os.makedirs(output_dir, exist_ok=True)
        indiv_log_dir = os.path.join(output_dir, 'individual_conversation_logs')
        os.makedirs(indiv_log_dir, exist_ok=True)
        log_file = os.path.join(indiv_log_dir, f"conversation_{question['id']}.log")
        with open(log_file, 'w', encoding='utf-8') as f:
            f.write(result.conversation_log)
        
        print(f"\nSingle Test Results for {args.test_single}:")
        print(f"Question: {result.static_question}")
        print(f"Agent Response: {result.agent_response}")
        print(f"Tools Used: {result.tools_used}")
        print(f"Score: {result.evaluation_score:.2f}")
        print(f"Response Time: {result.response_time:.2f}s")
        print(f"Results saved to: {output_dir}")
        
    else:
        # Run comprehensive evaluation
        results = await evaluator.run_comprehensive_evaluation(str(args.output_dir) if args.output_dir else None)
        
        # Print summary
        print(f"\nEvaluation Summary:")
        print(f"Total Questions: {results['total_questions']}")
        print(f"Average Score: {results['average_score']:.2f}")
        print(f"Average Response Time: {results['average_response_time']:.2f}s")
        print(f"Difficulty Scores: {results['difficulty_scores']}")
        print(f"Tool Usage: {results['tool_usage_stats']}")


if __name__ == "__main__":
    # No longer needed as logging is configured in __init__
    # logging.basicConfig(level=logging.INFO)
    asyncio.run(main()) 