#!/usr/bin/env python3
"""
Test script to verify max turns exceeded logic
"""

import asyncio
import os
import sys
from datetime import datetime

# Add the parent directory to the path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.conversation_evaluator import DynamicConversationEvaluator

async def test_max_turns():
    """Test the max turns exceeded logic"""
    print("Testing max turns exceeded logic...")
    
    # Create evaluator
    evaluator = DynamicConversationEvaluator()
    
    # Create a test question that might trigger multiple clarification requests
    test_question = {
        "id": "test_max_turns",
        "difficulty": "hard",
        "category": "network_analysis",
        "static_question": "Can you provide a comprehensive analysis of the network performance including all UEs, their current status, signal quality metrics, handover patterns, and any potential issues that might affect service quality? I need this for a detailed report.",
        "evaluation_criteria": {
            "should_use_tools": True,
            "self_evaluation_prompt": "Evaluate if the response provides comprehensive network analysis"
        },
        "expected_tools": ["get_knowledge"],
        "expected_response": "Comprehensive network analysis with multiple data points"
    }
    
    print(f"Testing question: {test_question['static_question']}")
    print("This question is designed to potentially trigger multiple clarification requests...")
    
    # Run evaluation
    result = await evaluator.run_conversation_evaluation(test_question)
    
    print(f"\nEvaluation completed!")
    print(f"Question ID: {result.question_id}")
    print(f"Total conversation turns: {len(result.dynamic_conversation)}")
    print(f"Final score: {result.evaluation_score:.2f}")
    print(f"Response time: {result.response_time:.2f}s")
    
    # Check if max turns were exceeded
    if len(result.dynamic_conversation) >= 10:
        print("✅ Max turns (10) were reached/exceeded as expected")
    else:
        print(f"ℹ️  Conversation completed in {len(result.dynamic_conversation)} turns (max: 10)")
    
    print(f"\nConversation log saved to: {result.conversation_log}")
    
    return result

if __name__ == "__main__":
    # Set OpenAI API key if not already set
    if not os.getenv("OPENAI_API_KEY"):
        print("Please set OPENAI_API_KEY environment variable")
        sys.exit(1)
    
    # Run the test
    asyncio.run(test_max_turns()) 