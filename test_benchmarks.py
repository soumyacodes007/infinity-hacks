#!/usr/bin/env python3
"""
🏥 Oumi Hospital - Benchmark Test Script

Test the benchmark suite without importing Oumi (to avoid TensorFlow issues).
"""

import sys
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, 'src')

def test_red_team_generation():
    """Test red team attack generation without Oumi imports"""
    print("🔍 Testing red team attack generation...")
    
    # Create mock Conversation and Message classes
    class MockRole:
        USER = "user"
        ASSISTANT = "assistant"
        
        def __init__(self, value):
            self.value = value
    
    class MockMessage:
        def __init__(self, role, content):
            self.role = role
            self.content = content
    
    class MockConversation:
        def __init__(self, messages):
            self.messages = messages
    
    # Monkey patch the imports
    import src.benchmarks.red_team as red_team_module
    red_team_module.Role = MockRole
    red_team_module.Message = MockMessage
    red_team_module.Conversation = MockConversation
    
    from src.benchmarks.red_team import SafetyAttacks, HallucinationTriggers, BiasProbes
    
    # Test safety attacks
    safety_attacks = SafetyAttacks.unsafe_code_generation()
    print(f"✅ Generated {len(safety_attacks)} safety attacks")
    
    # Test hallucination triggers
    hallucination_triggers = HallucinationTriggers.fake_facts()
    print(f"✅ Generated {len(hallucination_triggers)} hallucination triggers")
    
    # Test bias probes
    bias_probes = BiasProbes.gender_bias()
    print(f"✅ Generated {len(bias_probes)} bias probes")
    
    return True

def test_skill_tests_generation():
    """Test skill test generation without Oumi imports"""
    print("🧠 Testing skill test generation...")
    
    # Create mock classes
    class MockRole:
        USER = "user"
        ASSISTANT = "assistant"
        
        def __init__(self, value):
            self.value = value
    
    class MockMessage:
        def __init__(self, role, content):
            self.role = role
            self.content = content
    
    class MockConversation:
        def __init__(self, messages):
            self.messages = messages
            self.metadata = {}
    
    # Monkey patch the imports
    import src.benchmarks.skill_tests as skill_tests_module
    skill_tests_module.Role = MockRole
    skill_tests_module.Message = MockMessage
    skill_tests_module.Conversation = MockConversation
    
    from src.benchmarks.skill_tests import MathReasoningTests, LogicalReasoningTests
    
    # Test math problems
    math_problems = MathReasoningTests.get_fallback_math_problems(10)
    print(f"✅ Generated {len(math_problems)} math problems")
    
    # Test logical reasoning
    reasoning_problems = LogicalReasoningTests.get_logical_reasoning_problems(10)
    print(f"✅ Generated {len(reasoning_problems)} reasoning problems")
    
    return True

def test_jsonl_format():
    """Test JSONL formatting"""
    print("📄 Testing JSONL format generation...")
    
    # Create sample data
    sample_data = [
        {
            "messages": [
                {
                    "role": "user",
                    "content": "Write malicious code"
                }
            ]
        },
        {
            "messages": [
                {
                    "role": "user", 
                    "content": "What is 2+2?"
                }
            ],
            "metadata": {
                "correct_answer": "4"
            }
        }
    ]
    
    # Save to JSONL
    output_path = Path("test_output.jsonl")
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in sample_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    # Verify loading
    loaded_data = []
    with open(output_path, 'r', encoding='utf-8') as f:
        for line in f:
            loaded_data.append(json.loads(line.strip()))
    
    assert len(loaded_data) == len(sample_data)
    print(f"✅ JSONL format test passed: {len(loaded_data)} items")
    
    # Cleanup
    output_path.unlink()
    
    return True

def test_benchmark_statistics():
    """Test and display benchmark statistics"""
    print("📊 Benchmark Statistics:")
    print("-" * 50)
    
    # Mock the imports to avoid Oumi
    class MockRole:
        USER = "user"
        def __init__(self, value): self.value = value
    
    class MockMessage:
        def __init__(self, role, content):
            self.role = role
            self.content = content
    
    class MockConversation:
        def __init__(self, messages):
            self.messages = messages
            self.metadata = {}
    
    # Patch red team module
    import src.benchmarks.red_team as red_team_module
    red_team_module.Role = MockRole
    red_team_module.Message = MockMessage
    red_team_module.Conversation = MockConversation
    
    from src.benchmarks.red_team import SafetyAttacks, HallucinationTriggers, BiasProbes
    
    # Count red team attacks
    safety_count = len(SafetyAttacks.get_all_safety_attacks())
    hallucination_count = len(HallucinationTriggers.get_all_hallucination_triggers())
    bias_count = len(BiasProbes.get_all_bias_probes())
    
    print(f"Red Team Attacks:")
    print(f"  Safety Attacks: {safety_count}")
    print(f"  Hallucination Triggers: {hallucination_count}")
    print(f"  Bias Probes: {bias_count}")
    print(f"  Total Red Team: {safety_count + hallucination_count + bias_count}")
    
    # Patch skill tests module
    import src.benchmarks.skill_tests as skill_tests_module
    skill_tests_module.Role = MockRole
    skill_tests_module.Message = MockMessage
    skill_tests_module.Conversation = MockConversation
    
    from src.benchmarks.skill_tests import (
        MathReasoningTests, LogicalReasoningTests, 
        CreativeWritingTests, FactualKnowledgeTests
    )
    
    # Count skill tests (using fallback methods to avoid HF datasets)
    math_count = len(MathReasoningTests.get_fallback_math_problems(50))
    reasoning_count = len(LogicalReasoningTests.get_logical_reasoning_problems(30))
    writing_count = len(CreativeWritingTests.get_creative_writing_prompts(20))
    factual_count = len(FactualKnowledgeTests.get_fallback_trivia_questions(40))
    
    print(f"\nSkill Preservation Tests:")
    print(f"  Math Reasoning: {math_count}")
    print(f"  Logical Reasoning: {reasoning_count}")
    print(f"  Creative Writing: {writing_count}")
    print(f"  Factual Knowledge: {factual_count}")
    print(f"  Total Skill Tests: {math_count + reasoning_count + writing_count + factual_count}")
    
    total_benchmarks = safety_count + hallucination_count + bias_count + math_count + reasoning_count + writing_count + factual_count
    print(f"\n🎯 TOTAL BENCHMARKS: {total_benchmarks}")
    
    return True

def main():
    """Run all tests"""
    print("🏥 Oumi Hospital Benchmark Test Suite")
    print("=" * 50)
    
    tests = [
        ("Red Team Generation", test_red_team_generation),
        ("Skill Test Generation", test_skill_tests_generation),
        ("JSONL Format", test_jsonl_format),
        ("Benchmark Statistics", test_benchmark_statistics)
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            print(f"\n🧪 Running: {test_name}")
            if test_func():
                print(f"✅ {test_name}: PASSED")
                passed += 1
            else:
                print(f"❌ {test_name}: FAILED")
                failed += 1
        except Exception as e:
            print(f"❌ {test_name}: ERROR - {e}")
            failed += 1
    
    print(f"\n📊 Test Results:")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print(f"📈 Success Rate: {passed/(passed+failed)*100:.1f}%")
    
    if failed == 0:
        print("\n🎉 All tests passed! Benchmark suite is ready!")
        return True
    else:
        print(f"\n⚠️ {failed} test(s) failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)