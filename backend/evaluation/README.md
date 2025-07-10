# Dynamic Conversation Evaluator

This directory contains the dynamic conversation evaluator for the Network Engineer Chat Agent. The evaluator uses AI to generate realistic multi-turn conversations from static questions and evaluates agent responses based on tool outputs and AI self-evaluation.

## Files

- `conversation_data.json` - Static questions and evaluation criteria
- `conversation_evaluator.py` - Main evaluator implementation
- `test_conversation_evaluator.py` - Test script to demonstrate usage
- `setup_test.py` - Setup script to check requirements

## Features

### 1. Dynamic Conversation Generation
- Uses AI to generate natural follow-up questions from static questions
- Creates realistic multi-turn conversations as a network engineer would have
- Maintains conversation context and flow

### 2. Comprehensive Evaluation
- Evaluates responses by comparing with actual tool outputs when tools are used
- Uses AI self-evaluation when no tools are used
- Provides detailed scoring and reasoning for each response
- **Improved scoring** - No more generic 0.5 scores, provides specific detailed evaluations

### 3. Detailed Logging
- **Individual conversation logs** - Each question gets its own detailed log file
- **Complete conversation tracking** - Logs every turn, tool usage, and response
- **Debugging information** - Easy to understand what happened in each conversation

### 4. Organized Results
- **Timestamped folders** - Results saved in folders with current timestamp
- **Multiple output formats** - JSON reports, summary files, and text logs
- **Easy analysis** - All results organized for easy review

### 5. Simulation Integration
- **Proper simulation initialization** - Starts simulation like the frontend does
- **Real network state** - Evaluates against actual running simulation
- **Knowledge layer integration** - Uses real knowledge router and tools

## How to Test

### Prerequisites

1. Make sure you have the required dependencies installed:
```bash
pip install -r requirements.txt
```

2. Set up your OpenAI API key:
```bash
export OPENAI_API_KEY="your-api-key-here"
```

3. Ensure the backend simulation is running or the knowledge layer is properly initialized.

### Running the Tests

#### Option 1: Run the Test Script
```bash
cd backend/evaluation
python test_conversation_evaluator.py
```

This will run:
1. **Simulation initialization test** - Verifies simulation starts properly
2. **Conversation generation test** - Shows how AI creates dynamic conversations
3. **Single question evaluation** - Tests one question with full evaluation and logging
4. **Comprehensive evaluation** - Runs all 15 questions and generates organized reports

#### Option 2: Run Individual Components

**Test simulation initialization:**
```python
from conversation_evaluator import DynamicConversationEvaluator
import asyncio

async def test():
    evaluator = DynamicConversationEvaluator()
    evaluator._initialize_simulation()
    print("Simulation initialized successfully")

asyncio.run(test())
```

**Test conversation generation:**
```python
from conversation_evaluator import DynamicConversationEvaluator
import asyncio

async def test():
    evaluator = DynamicConversationEvaluator()
    conversation = await evaluator.generate_dynamic_conversation(
        "What is the current status of user equipment UE_001?"
    )
    print(conversation)

asyncio.run(test())
```

**Test single question evaluation:**
```python
from conversation_evaluator import DynamicConversationEvaluator
import asyncio

async def test():
    evaluator = DynamicConversationEvaluator()
    
    # Initialize simulation first
    evaluator._initialize_simulation()
    
    test_question = {
        "id": "test_001",
        "difficulty": "easy",
        "category": "ue_status",
        "static_question": "What is the current status of user equipment UE_001?",
        "expected_tools": ["get_knowledge"],
        "evaluation_criteria": {
            "should_use_tools": True,
            "expected_tool_output_contains": ["UE_001", "status"],
            "self_evaluation_prompt": "Evaluate if the response accurately describes UE_001's status"
        }
    }
    
    result = await evaluator.run_conversation_evaluation(test_question)
    print(f"Score: {result.evaluation_score}")
    print(f"Reasoning: {result.evaluation_reasoning}")
    
    # Save conversation log
    with open("test_conversation.log", "w") as f:
        f.write(result.conversation_log)

asyncio.run(test())
```

**Run comprehensive evaluation:**
```python
from conversation_evaluator import DynamicConversationEvaluator
import asyncio

async def test():
    evaluator = DynamicConversationEvaluator()
    results = await evaluator.run_comprehensive_evaluation()
    print(f"Average Score: {results['average_score']}")

asyncio.run(test())
```

### Option 3: Command Line Interface

Run the evaluator directly with command line arguments:

```bash
cd backend/evaluation
python conversation_evaluator.py --output-dir my_results --conversation-data conversation_data.json
```

## Output Structure

After running the evaluation, you'll find a timestamped folder (e.g., `evaluation_results_20240115_143022/`) containing:

### Files Generated

1. **Individual conversation logs** - `conversation_q_001.log`, `conversation_q_002.log`, etc.
   - Detailed logs for each question showing the full conversation flow
   - Tool usage and outputs
   - Evaluation reasoning

2. **Comprehensive evaluation report** - `dynamic_conversation_evaluation_YYYYMMDD_HHMMSS.json`
   - Complete evaluation results with all details
   - Individual scores and reasoning for each question

3. **Summary report** - `evaluation_summary_YYYYMMDD_HHMMSS.json`
   - High-level statistics and scores by category/difficulty

4. **All conversation logs** - `all_conversation_logs_YYYYMMDD_HHMMSS.txt`
   - Combined log file with all conversations for easy review

### Sample Output Structure

```
evaluation_results_20240115_143022/
├── conversation_q_001.log
├── conversation_q_002.log
├── conversation_q_003.log
├── ...
├── dynamic_conversation_evaluation_20240115_143022.json
├── evaluation_summary_20240115_143022.json
└── all_conversation_logs_20240115_143022.txt
```

### Sample Conversation Log

```
=== CONVERSATION LOG FOR q_001 ===
Timestamp: 2024-01-15T14:30:22.123456
Static Question: What is the current status of user equipment UE_001?

DYNAMIC CONVERSATION:
Turn 1: What is the current status of user equipment UE_001?
Turn 2: Can you also tell me about its connection quality and which cell it's connected to?
Turn 3: What about its performance metrics?

AGENT RESPONSE:
Based on the current simulation state, UE_001 is connected to cell BS_001_Cell_1...

TOOLS USED: ['get_knowledge']

TOOL OUTPUTS:
get_knowledge: {"ue_id": "UE_001", "status": "connected", "cell": "BS_001_Cell_1"}

RESPONSE TIME: 2.34s
EVALUATION SCORE: 0.85
EVALUATION REASONING:
The agent correctly used the get_knowledge tool and provided accurate information...

=== END LOG ===
```

### Sample Summary Report

```json
{
  "evaluation_timestamp": "2024-01-15T14:30:22",
  "total_questions": 15,
  "average_score": 0.82,
  "average_response_time": 2.1,
  "difficulty_scores": {
    "easy": 0.88,
    "medium": 0.81,
    "hard": 0.76
  },
  "category_scores": {
    "ue_status": 0.90,
    "network_overview": 0.88,
    "ai_services": 0.85
  },
  "tool_usage_stats": {
    "get_knowledge": 12,
    "get_knowledge_bulk": 8
  }
}
```

## Key Improvements

### 1. **Detailed Logging**
- Each question gets its own log file
- Complete conversation flow tracking
- Tool usage and outputs logged
- Evaluation reasoning preserved

### 2. **Organized Results**
- Timestamped folders for each evaluation run
- Multiple output formats for different analysis needs
- Easy to track progress and compare results

### 3. **Simulation Integration**
- Proper simulation initialization like the frontend
- Real network state for more accurate evaluation
- Knowledge layer properly integrated

## Customizing Questions

To add or modify questions, edit `conversation_data.json`:

```json
{
  "id": "q_new",
  "difficulty": "medium",
  "category": "custom_category",
  "description": "Your question description",
  "static_question": "Your question here?",
  "expected_tools": ["get_knowledge"],
  "expected_agent": "Basic Network Knowledge Assistant",
  "evaluation_criteria": {
    "should_use_tools": true,
    "expected_tool_output_contains": ["expected", "keywords"],
    "self_evaluation_prompt": "Your evaluation prompt"
  }
}
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Make sure you're running from the correct directory and the backend modules are accessible.

2. **API Key Issues**: Ensure your OpenAI API key is set correctly:
```bash
export OPENAI_API_KEY="your-key-here"
```

3. **Simulation Issues**: The evaluator now properly initializes simulation. If you get simulation errors, check that all backend dependencies are installed.

4. **Memory Issues**: For large evaluations, consider running fewer questions at a time or increasing your system's memory allocation.

### Debug Mode

To see detailed logging, set the log level:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Evaluation Metrics

The evaluator provides several metrics:

- **Accuracy Score**: How well the response matches expected tool outputs
- **Tool Usage**: Whether the agent used appropriate tools
- **Response Time**: How quickly the agent responds
- **Completeness**: Whether the response addresses the full question
- **Clarity**: How clear and understandable the response is

## Extending the Evaluator

To add new evaluation criteria or modify the evaluation logic:

1. Edit the `_create_evaluation_ai()` method in `conversation_evaluator.py`
2. Modify the evaluation prompts to include your new criteria
3. Update the score parsing logic if needed

To add new conversation generation patterns:

1. Edit the `_create_conversation_ai()` method
2. Modify the conversation generation prompts
3. Update the conversation parsing logic if needed 