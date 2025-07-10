#!/usr/bin/env python3
"""
Simple script to run the Dynamic Conversation Evaluator
"""

import asyncio
import logging
import sys
import os

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from conversation_evaluator import DynamicConversationEvaluator

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


async def main():
    """Run the evaluation"""
    print("🚀 Starting Dynamic Conversation Evaluator")
    print("=" * 50)
    
    try:
        # Create evaluator
        evaluator = DynamicConversationEvaluator()
        
        # Run comprehensive evaluation
        print("📊 Running comprehensive evaluation...")
        results = await evaluator.run_comprehensive_evaluation()
        
        if "error" in results:
            print(f"❌ Error: {results['error']}")
            return
        
        # Print summary
        print("\n✅ Evaluation completed successfully!")
        print(f"📈 Total Questions: {results['total_questions']}")
        print(f"📊 Average Score: {results['average_score']:.2f}")
        print(f"⏱️  Average Response Time: {results['average_response_time']:.2f}s")
        print(f"🎯 Difficulty Scores: {results['difficulty_scores']}")
        print(f"🔧 Tool Usage: {results['tool_usage_stats']}")
        
        # Find the output directory
        import glob
        output_dirs = glob.glob("evaluation_results_*")
        if output_dirs:
            latest_dir = max(output_dirs, key=os.path.getctime)
            print(f"\n📁 Results saved to: {latest_dir}")
            print(f"📄 Check the following files:")
            print(f"   - Individual conversation logs: conversation_*.log")
            print(f"   - Complete evaluation: dynamic_conversation_evaluation_*.json")
            print(f"   - Summary report: evaluation_summary_*.json")
            print(f"   - All logs combined: all_conversation_logs_*.txt")
        
    except Exception as e:
        print(f"❌ Error running evaluation: {e}")
        logger.error(f"Evaluation failed: {e}", exc_info=True)


if __name__ == "__main__":
    asyncio.run(main()) 