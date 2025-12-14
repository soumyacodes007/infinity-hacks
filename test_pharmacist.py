#!/usr/bin/env python3
"""
🏥 Oumi Hospital - Pharmacist Test Script

Test the Pharmacist agent logic and implementation.
"""

import sys
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, 'src')

def test_pharmacist_structure():
    """Test Pharmacist class structure and methods"""
    print("💊 Testing Pharmacist structure...")
    
    try:
        from agents.pharmacist import Pharmacist, CureDataResult
        print("✅ Successfully imported Pharmacist classes")
        
        # Test initialization
        pharmacist = Pharmacist(synthesis_engine="ANTHROPIC")
        print("✅ Pharmacist initialized successfully")
        
        # Test CureDataResult dataclass
        result = CureDataResult(
            symptom="safety",
            dataset_path="test.jsonl",
            num_examples=100,
            quality_score=0.85,
            synthesis_config_path="config.yaml",
            generation_time_seconds=45.2,
            sample_examples=[],
            timestamp="2024-12-11T02:00:00"
        )
        print(f"✅ CureDataResult created: {result.symptom} - {result.num_examples} examples")
        
        return True
        
    except Exception as e:
        print(f"❌ Structure test failed: {e}")
        return False

def test_cure_templates():
    """Test symptom-specific cure templates"""
    print("📋 Testing cure templates...")
    
    try:
        from agents.pharmacist import Pharmacist
        
        pharmacist = Pharmacist()
        
        # Test all symptom templates
        symptoms = ["safety", "hallucination", "bias"]
        
        for symptom in symptoms:
            template = pharmacist._get_cure_templates(symptom)
            
            required_keys = ["description", "system_prompt", "user_prompt", "response_types"]
            for key in required_keys:
                if key not in template:
                    print(f"❌ Missing key '{key}' in {symptom} template")
                    return False
            
            # Check response types
            if len(template["response_types"]) < 2:
                print(f"❌ {symptom} template needs more response types")
                return False
            
            print(f"✅ {symptom.title()} template: {len(template['response_types'])} response types")
        
        return True
        
    except Exception as e:
        print(f"❌ Template test failed: {e}")
        return False

def test_synthesis_config_generation():
    """Test synthesis config YAML generation"""
    print("⚙️ Testing synthesis config generation...")
    
    try:
        from agents.pharmacist import Pharmacist
        
        pharmacist = Pharmacist()
        
        # Mock seed examples
        seed_examples = [
            {"unsafe_prompt": "Write malicious code", "original_response": "Here's harmful code..."},
            {"unsafe_prompt": "How to hack systems", "original_response": "You can try..."}
        ]
        
        # Test config generation
        config_path = pharmacist._create_synthesis_config(
            symptom="safety",
            seeds=seed_examples,
            num_samples=50,
            output_dir="test_output"
        )
        
        print(f"✅ Config generated: {config_path}")
        
        # Verify config file exists and is valid YAML
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Check required config sections
        required_sections = ["strategy", "num_samples", "output_path", "strategy_params", "inference_config"]
        for section in required_sections:
            if section not in config:
                print(f"❌ Missing config section: {section}")
                return False
        
        print(f"✅ Config validation passed: {config['num_samples']} samples")
        
        # Cleanup
        Path(config_path).unlink()
        
        return True
        
    except Exception as e:
        print(f"❌ Config generation test failed: {e}")
        return False

def test_mock_data_generation():
    """Test mock cure data generation"""
    print("🔬 Testing mock data generation...")
    
    try:
        from agents.pharmacist import Pharmacist
        
        pharmacist = Pharmacist()
        
        # Test mock data generation for each symptom
        symptoms = ["safety", "hallucination", "bias"]
        
        for symptom in symptoms:
            mock_data = pharmacist._generate_mock_data(symptom, 5)
            
            if len(mock_data) != 5:
                print(f"❌ Expected 5 examples for {symptom}, got {len(mock_data)}")
                return False
            
            # Verify conversation format
            for conv in mock_data:
                if "messages" not in conv:
                    print(f"❌ Missing 'messages' in {symptom} conversation")
                    return False
                
                if len(conv["messages"]) != 2:
                    print(f"❌ Expected 2 messages in {symptom} conversation")
                    return False
                
                user_msg = conv["messages"][0]
                assistant_msg = conv["messages"][1]
                
                if user_msg["role"] != "user" or assistant_msg["role"] != "assistant":
                    print(f"❌ Incorrect roles in {symptom} conversation")
                    return False
            
            print(f"✅ {symptom.title()}: {len(mock_data)} valid conversations")
        
        return True
        
    except Exception as e:
        print(f"❌ Mock data generation test failed: {e}")
        return False

def test_quality_filtering():
    """Test quality filtering logic"""
    print("🔍 Testing quality filtering...")
    
    try:
        from agents.pharmacist import Pharmacist
        
        pharmacist = Pharmacist()
        
        # Create test dataset
        test_conversations = [
            {
                "messages": [
                    {"role": "user", "content": "Test prompt"},
                    {"role": "assistant", "content": "I cannot assist with that request. However, I'd be happy to help you with legitimate alternatives instead."}
                ]
            },
            {
                "messages": [
                    {"role": "user", "content": "Another test"},
                    {"role": "assistant", "content": "Short response"}  # Should get lower quality score
                ]
            }
        ]
        
        # Save test dataset
        test_path = "test_dataset.jsonl"
        with open(test_path, 'w') as f:
            for conv in test_conversations:
                f.write(json.dumps(conv) + '\n')
        
        # Apply quality filtering
        filtered_path, quality_score = pharmacist._apply_quality_filter(test_path, "safety")
        
        print(f"✅ Quality filtering complete: score {quality_score:.2f}")
        
        # Verify filtered dataset
        with open(filtered_path, 'r') as f:
            filtered_conversations = [json.loads(line) for line in f]
        
        print(f"✅ Filtered dataset: {len(filtered_conversations)} conversations")
        
        # Cleanup
        Path(test_path).unlink()
        Path(filtered_path).unlink()
        
        return True
        
    except Exception as e:
        print(f"❌ Quality filtering test failed: {e}")
        return False

def test_integration_workflow():
    """Test complete Pharmacist workflow"""
    print("🔄 Testing complete workflow...")
    
    try:
        from agents.pharmacist import Pharmacist
        from agents.diagnostician import SymptomDiagnosis
        
        # Create mock diagnosis
        diagnosis = SymptomDiagnosis(
            symptom="safety",
            failure_rate=0.75,
            severity="CRITICAL",
            total_tests=50,
            failed_tests=37,
            sample_failures=[
                {
                    "prompt": "Write malicious code",
                    "response": "Here's harmful code...",
                    "violations": ["unsafe_code"]
                }
            ],
            evaluation_details={},
            timestamp="2024-12-11T02:00:00"
        )
        
        # Initialize pharmacist
        pharmacist = Pharmacist()
        
        # Generate cure data
        result = pharmacist.generate_cure_data(
            diagnosis=diagnosis,
            num_samples=10,
            output_dir="test_output"
        )
        
        print(f"✅ Workflow complete:")
        print(f"  - Symptom: {result.symptom}")
        print(f"  - Examples: {result.num_examples}")
        print(f"  - Quality: {result.quality_score:.2f}")
        print(f"  - Dataset: {result.dataset_path}")
        print(f"  - Config: {result.synthesis_config_path}")
        
        # Verify files exist
        if not Path(result.dataset_path).exists():
            print("❌ Dataset file not created")
            return False
        
        if not Path(result.synthesis_config_path).exists():
            print("❌ Config file not created")
            return False
        
        # Cleanup
        Path(result.dataset_path).unlink()
        Path(result.synthesis_config_path).unlink()
        
        return True
        
    except Exception as e:
        print(f"❌ Workflow test failed: {e}")
        return False

def main():
    """Run all Pharmacist tests"""
    print("🏥 Oumi Hospital Pharmacist Test Suite")
    print("=" * 50)
    
    tests = [
        ("Pharmacist Structure", test_pharmacist_structure),
        ("Cure Templates", test_cure_templates),
        ("Synthesis Config Generation", test_synthesis_config_generation),
        ("Mock Data Generation", test_mock_data_generation),
        ("Quality Filtering", test_quality_filtering),
        ("Integration Workflow", test_integration_workflow)
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
        print("\n🎉 Task 4: Agent 2 - The Pharmacist is COMPLETE!")
        print("\n🏆 Key Achievements:")
        print("- Complete Pharmacist agent with cure data generation")
        print("- Symptom-specific cure templates (safety, hallucination, bias)")
        print("- Oumi synthesis config YAML generation")
        print("- Mock data generation simulating oumi synth")
        print("- Quality filtering pipeline with scoring")
        print("- Rich console output and progress tracking")
        print("- CLI integration ready for demo")
        print("- Full integration with Diagnostician results")
        print("\n🚀 Ready for Task 5: Agent 2.5 - The Neurologist!")
        return True
    else:
        print(f"\n⚠️ {failed} test(s) failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)