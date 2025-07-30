"""
Test script for Dynamic Conversation Evaluator
"""

import asyncio
import logging
import sys
import os

# Add the backend directory to the path so we can import modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from conversation_evaluator import DynamicConversationEvaluator

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


async def test_single_question():
    """Test evaluation with a single question"""
    logger.info("Testing single question evaluation...")
    
    evaluator = DynamicConversationEvaluator()
    
    # Initialize simulation first
    evaluator._initialize_simulation()
    
    # Test with a simple question
    test_question = {
        "id": "test_001",
        "difficulty": "easy",
        "category": "ue_status",
        "description": "Test UE status query",
        "static_question": "What is the current status of user equipment UE_001?",
        "expected_tools": ["get_knowledge"],
        "expected_agent": "Basic Network Knowledge Assistant",
        "evaluation_criteria": {
            "should_use_tools": True,
            "expected_tool_output_contains": ["UE_001", "status", "connected"],
            "self_evaluation_prompt": "Evaluate if the response accurately describes UE_001's current status"
        }
    }
    
    result = await evaluator.run_conversation_evaluation(test_question)
    
    print(f"\nTest Result:")
    print(f"Question ID: {result.question_id}")
    print(f"Static Question: {result.static_question}")
    print(f"Dynamic Conversation: {result.dynamic_conversation}")
    print(f"Agent Response: {result.agent_response[:200]}...")
    print(f"Tools Used: {result.tools_used}")
    print(f"Response Time: {result.response_time:.2f}s")
    print(f"Evaluation Score: {result.evaluation_score:.2f}")
    print(f"Evaluation Reasoning: {result.evaluation_reasoning[:200]}...")
    
    # Save the conversation log
    with open("test_conversation.log", "w", encoding="utf-8") as f:
        f.write(result.conversation_log)
    print(f"Conversation log saved to test_conversation.log")


async def test_comprehensive_evaluation():
    """Test comprehensive evaluation with all questions"""
    logger.info("Testing comprehensive evaluation...")
    
    evaluator = DynamicConversationEvaluator()
    
    # Run comprehensive evaluation (will create timestamped folder)
    results = await evaluator.run_comprehensive_evaluation()
    
    if "error" in results:
        print(f"Error: {results['error']}")
        return
    
    print(f"\nComprehensive Evaluation Results:")
    print(f"Total Questions: {results['total_questions']}")
    print(f"Average Score: {results['average_score']:.2f}")
    print(f"Average Response Time: {results['average_response_time']:.2f}s")
    print(f"Difficulty Scores: {results['difficulty_scores']}")
    print(f"Tool Usage Stats: {results['tool_usage_stats']}")
    
    # Show some detailed results
    print(f"\nSample Detailed Results:")
    for i, result in enumerate(results['detailed_results'][:3]):  # Show first 3 results
        print(f"\nResult {i+1}:")
        print(f"  Question: {result['static_question']}")
        print(f"  Score: {result['evaluation_score']:.2f}")
        print(f"  Tools Used: {result['tools_used']}")
        print(f"  Response Time: {result['response_time']:.2f}s")


async def test_simulation_initialization():
    """Test simulation initialization"""
    logger.info("Testing simulation initialization...")
    
    evaluator = DynamicConversationEvaluator()
    
    try:
        evaluator._initialize_simulation()
        print("✓ Simulation initialized successfully")
        
        # Check if simulation engine is properly set up
        if evaluator.simulation_engine:
            print(f"✓ Simulation engine created with {len(evaluator.simulation_engine.ue_list)} UEs")
        else:
            print("✗ Simulation engine not created")
            
        if evaluator.knowledge_router:
            print("✓ Knowledge router initialized")
        else:
            print("✗ Knowledge router not initialized")
            
    except Exception as e:
        print(f"✗ Simulation initialization failed: {e}")


async def main():
    """Main test function"""
    print("Dynamic Conversation Evaluator Test")
    print("=" * 50)
    
    # Test simulation initialization
    print("\n1. Testing simulation initialization...")
    await test_simulation_initialization()
    
    # Test single question evaluation
    print("\n2. Testing single question evaluation...")
    await test_single_question()
    
    # Test comprehensive evaluation
    print("\n3. Testing comprehensive evaluation...")
    await test_comprehensive_evaluation()
    
    print("\nTest completed! Check the evaluation_results_* directory for output files.")


if __name__ == "__main__":
    asyncio.run(main()) 