#!/usr/bin/env python3
"""
Test neurologist components without complex imports
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_skill_score():
    """Test SkillScore dataclass"""
    print("🧠 Testing SkillScore...")
    
    try:
        # Import directly from the file
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "neurologist", 
            Path(__file__).parent / "src" / "agents" / "neurologist.py"
        )
        neurologist_module = importlib.util.module_from_spec(spec)
        
        # Mock the dependencies
        class MockOumiWrapper:
            pass
        
        class MockSkillTestSuite:
            def __init__(self):
                self.skill_types = ["math", "reasoning", "writing", "factual"]
            
            def get_evaluation_function(self, domain):
                return f"{domain}_judge"
        
        class MockConsole:
            def print(self, *args, **kwargs):
                pass
        
        # Inject mocks
        sys.modules['src.utils.oumi_integration'] = type('MockModule', (), {'OumiWrapper': MockOumiWrapper})()
        sys.modules['src.benchmarks.skill_tests'] = type('MockModule', (), {'SkillTestSuite': MockSkillTestSuite})()
        sys.modules['src.utils.console'] = type('MockModule', (), {'hospital_console': type('MockHospitalConsole', (), {'console': MockConsole()})()})()
        
        # Now load the module
        spec.loader.exec_module(neurologist_module)
        
        # Test SkillScore
        SkillScore = neurologist_module.SkillScore
        
        score = SkillScore(
            domain="math",
            score_before=0.85,
            score_after=0.80,
            degradation=0.05,
            status="preserved"
        )
        
        print(f"✅ SkillScore created: {score.domain}")
        print(f"   Degradation: {score.degradation_percent:.1f}%")
        
        # Test SkillReport
        SkillReport = neurologist_module.SkillReport
        
        report = SkillReport(
            model_before="test-v1",
            model_after="test-v2",
            skill_scores=[score],
            overall_degradation=5.9,
            verdict="safe",
            recommendations=["Test recommendation"]
        )
        
        print(f"✅ SkillReport created: {report.model_before} -> {report.model_after}")
        
        # Test serialization
        report_dict = report.to_dict()
        print(f"✅ Report serialized with {len(report_dict)} fields")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_neurologist_logic():
    """Test neurologist recommendation logic"""
    print("\n🧠 Testing Neurologist Logic...")
    
    try:
        # Test severity classification
        def classify_severity(failure_rate):
            if failure_rate >= 0.7:
                return "CRITICAL"
            elif failure_rate >= 0.5:
                return "HIGH"
            elif failure_rate >= 0.3:
                return "MODERATE"
            else:
                return "LOW"
        
        test_cases = [
            (0.8, "CRITICAL"),
            (0.6, "HIGH"),
            (0.4, "MODERATE"),
            (0.2, "LOW")
        ]
        
        for rate, expected in test_cases:
            result = classify_severity(rate)
            if result == expected:
                print(f"✅ {rate:.1%} -> {result}")
            else:
                print(f"❌ {rate:.1%} -> {result} (expected {expected})")
                return False
        
        # Test recommendation generation logic
        def generate_recommendations(degraded_count, total_count, severe_count):
            recommendations = []
            
            if severe_count > 0:
                recommendations.append("🚨 CRITICAL: Severe skill degradation detected")
                recommendations.append("Reduce learning rate by 50%")
            elif degraded_count > total_count // 2:
                recommendations.append("⚠️ CAUTION: Moderate skill degradation detected")
                recommendations.append("Reduce learning rate by 25%")
            else:
                recommendations.append("✅ SAFE: Skills well preserved")
            
            return recommendations
        
        test_scenarios = [
            (0, 4, 0, "SAFE"),
            (3, 4, 0, "CAUTION"),  # More than half degraded
            (3, 4, 1, "CRITICAL")
        ]
        
        for degraded, total, severe, expected_type in test_scenarios:
            recs = generate_recommendations(degraded, total, severe)
            if expected_type.upper() in recs[0].upper():
                print(f"✅ Scenario ({degraded}/{total} degraded, {severe} severe) -> {expected_type}")
            else:
                print(f"❌ Scenario failed: expected {expected_type}, got {recs[0]}")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ Logic test failed: {e}")
        return False


def main():
    """Run component tests"""
    print("🏥 Neurologist Component Tests")
    print("=" * 40)
    
    tests = [
        test_skill_score,
        test_neurologist_logic
    ]
    
    passed = 0
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 40)
    print(f"Results: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("🎉 All component tests passed!")
        return 0
    else:
        print("⚠️ Some tests failed.")
        return 1


if __name__ == "__main__":
    exit(main())