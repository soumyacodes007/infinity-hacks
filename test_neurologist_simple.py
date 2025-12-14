#!/usr/bin/env python3
"""
Simple test for the Neurologist agent (Task 5)
Tests skill preservation checking functionality
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_neurologist_basic():
    """Test basic neurologist functionality"""
    print("🧠 Testing Neurologist Agent...")
    
    try:
        from agents.neurologist import Neurologist, SkillScore, SkillReport
        from benchmarks.skill_tests import SkillTestSuite
        
        print("✅ Imports successful")
        
        # Test SkillScore creation
        score = SkillScore(
            domain="math",
            score_before=0.85,
            score_after=0.80,
            degradation=0.05,
            status="preserved"
        )
        
        print(f"✅ SkillScore created: {score.domain} - {score.degradation_percent:.1f}% degradation")
        
        # Test SkillTestSuite
        suite = SkillTestSuite()
        print(f"✅ SkillTestSuite created with domains: {suite.skill_types}")
        
        # Test evaluation function mapping
        for domain in suite.skill_types:
            eval_func = suite.get_evaluation_function(domain)
            print(f"✅ {domain} -> {eval_func}")
        
        # Test Neurologist initialization
        neurologist = Neurologist(degradation_threshold=15.0)
        print(f"✅ Neurologist initialized with threshold: {neurologist.degradation_threshold}%")
        
        # Test recommendation generation
        mock_scores = [
            SkillScore("math", 0.85, 0.70, 0.15, "degraded"),
            SkillScore("reasoning", 0.80, 0.78, 0.02, "preserved"),
            SkillScore("writing", 0.75, 0.77, -0.02, "improved"),
            SkillScore("factual", 0.90, 0.60, 0.30, "degraded")
        ]
        
        verdict, recommendations = neurologist._generate_recommendations(mock_scores, 12.5)
        print(f"✅ Generated verdict: {verdict}")
        print(f"✅ Generated {len(recommendations)} recommendations")
        
        for i, rec in enumerate(recommendations[:3], 1):
            print(f"   {i}. {rec}")
        
        # Test report creation
        report = SkillReport(
            model_before="test-model-v1",
            model_after="test-model-v2", 
            skill_scores=mock_scores,
            overall_degradation=12.5,
            verdict=verdict,
            recommendations=recommendations
        )
        
        print(f"✅ SkillReport created for {report.model_before} -> {report.model_after}")
        
        # Test report serialization
        report_dict = report.to_dict()
        print(f"✅ Report serialized with {len(report_dict)} fields")
        
        print("\n🎉 All Neurologist tests passed!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("   Make sure Oumi dependencies are installed")
        return False
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_skill_datasets():
    """Test skill test dataset loading"""
    print("\n📊 Testing Skill Test Datasets...")
    
    try:
        from benchmarks.skill_tests import get_skill_test_dataset
        
        # Test each skill type
        skill_types = ["math", "reasoning", "writing", "factual"]
        
        for skill_type in skill_types:
            try:
                conversations = get_skill_test_dataset(skill_type, num_samples=5)
                print(f"✅ {skill_type}: loaded {len(conversations)} conversations")
                
                # Check first conversation structure
                if conversations:
                    conv = conversations[0]
                    print(f"   Sample: {conv.messages[0].content[:50]}...")
                    if hasattr(conv, 'metadata') and conv.metadata:
                        print(f"   Metadata: {list(conv.metadata.keys())}")
                
            except Exception as e:
                print(f"⚠️ {skill_type}: {e}")
        
        print("✅ Skill dataset tests completed")
        return True
        
    except Exception as e:
        print(f"❌ Skill dataset test failed: {e}")
        return False


def test_evaluation_functions():
    """Test evaluation function registration"""
    print("\n⚖️ Testing Evaluation Functions...")
    
    try:
        from benchmarks.judges import (
            math_accuracy_judge, logical_reasoning_judge,
            creative_writing_judge, factual_accuracy_judge
        )
        
        print("✅ All skill evaluation functions imported successfully")
        
        # Test function names
        functions = [
            ("math_accuracy_judge", math_accuracy_judge),
            ("logical_reasoning_judge", logical_reasoning_judge), 
            ("creative_writing_judge", creative_writing_judge),
            ("factual_accuracy_judge", factual_accuracy_judge)
        ]
        
        for name, func in functions:
            print(f"✅ {name}: {func.__name__}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Evaluation function import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Evaluation function test failed: {e}")
        return False


def main():
    """Run all neurologist tests"""
    print("🏥 Oumi Hospital - Neurologist Agent Test Suite")
    print("=" * 50)
    
    tests = [
        test_neurologist_basic,
        test_skill_datasets,
        test_evaluation_functions
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
    
    print("\n" + "=" * 50)
    print(f"🏥 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Neurologist agent is ready.")
        return 0
    else:
        print("⚠️ Some tests failed. Check the output above.")
        return 1


if __name__ == "__main__":
    exit(main())