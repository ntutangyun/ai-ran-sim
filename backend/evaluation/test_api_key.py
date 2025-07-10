#!/usr/bin/env python3
"""
Test script to verify OpenAI API key is working
"""

import os
import sys
import asyncio

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Set up OpenAI API key
from agents import set_default_openai_key, set_tracing_disabled

api_key = os.environ.get("OPENAI_API_KEY", "")
if not api_key:
    print("❌ OPENAI_API_KEY environment variable is not set")
    print("Please set it with: export OPENAI_API_KEY='your-api-key-here'")
    sys.exit(1)

print(f"✅ API Key found: {api_key[:10]}...")

set_default_openai_key(api_key)
set_tracing_disabled(True)

from agents import Agent, Runner

async def test_api_key():
    """Test if the API key works with a simple agent call"""
    try:
        # Create a simple test agent
        test_agent = Agent(
            name="Test Agent",
            instructions="You are a test agent. Just respond with 'Hello, API key is working!'",
            model="gpt-3.5-turbo"
        )
        
        print("🧪 Testing API key with a simple agent call...")
        
        # Make a simple call
        runner = Runner.run_streamed(test_agent, "Hello")
        
        response = ""
        async for event in runner.stream_events():
            if event.type == "message_output_item":
                response += event.data
        
        print(f"✅ API Key is working! Response: {response}")
        return True
        
    except Exception as e:
        print(f"❌ API Key test failed: {e}")
        return False

if __name__ == "__main__":
    asyncio.run(test_api_key()) 