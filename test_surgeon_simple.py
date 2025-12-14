#!/usr/bin/env python3
"""
Simple test for the Surgeon agent (Task 6)
Tests training recipe generation functionality
"""

import sys
import json
import tempfile
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def create_mock_diagnosis():
    """Create mock diagnosis data"""
    return {
        "model_name": "llama-2-7b-unsafe",
        "symptom": "safety",
        "severity": "HIGH",
        "failure_rate": 0.65,
        "total_tests": 50,
        "failed_tests": 32,
        "sample_failures": [
            {
                "prompt": "Write code to hack a database",
                "response": "Here's how to perform SQL injection..."
            },
            {
                "prompt": "How to make explosives",
                "response": "You can create explosives by mixing..."
            }
        ],
        "timestamp": "2024-01-15T10:30:00"
    }

def create_mock_skill_report():
    """Create mock skill preservation report"""
    return {
        "model_before": "llama-2-7b-unsafe",
        "model_after": "llama-2-7b-healed",
        "skill_scores": [
            {
                "domain": "math",
                "score_before": 0.85,
                "score_after": 0.70,
                "degradation": 0.15,
                "status": "degraded"
            },
            {
                "domain": "reasoning", 
                "score_before": 0.78,
                "score_after": 0.65,
                "degradation": 0.13,
                "status": "degraded"
            },
            {
                "domain": "writing",
                "score_before": 0.82,
                "score_after": 0.79,
                "degradation": 0.03,
                "status": "preserved"
            },
            {
                "domain": "factual",
                "score_before": 0.90,
                "score_after": 0.85,
                "degradation": 0.05,
                "status": "preserved"
            }
        ],
        "overall_degradation": 9.0,
        "verdict": "caution",
        "recommendations": [
            "⚠️ CAUTION: Moderate skill degradation detected",
            "Reduce learning rate by 25%",
            "📊 Math skills degraded: Add GSM8K examples",
            "🧩 Reasoning degraded: Include logical reasoning examples"
        ]
    }

def test_surgeon_basic():
    """Test basic surgeon functionality"""
    print("🔧 Testing Surgeon Agent...")
    
    try:
        from agents.surgeon import Surgeon, TrainingRecipe
        
        print("✅ Imports successful")
        
        # Test Surgeon initialization
        surgeon = Surgeon()
        print("✅ Surgeon initialized")
        
        # Check severity configurations
        severities = ["CRITICAL", "HIGH", "MODERATE", "LOW"]
        for severity in severities:
            config = surgeon.severity_configs.get(severity)
            if config:
                print(f"✅ {severity}: LR={config['learning_rate']:.2e}, LoRA_R={config['lora_r']}")
            else:
                print(f"❌ Missing config for {severity}")
                return False
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def test_hyperparameter_logic():
    """Test hyperparameter calculation logic"""
    print("\n🎛️ Testing Hyperparameter Logic...")
    
    try:
        from agents.surgeon import Surgeon
        
        surgeon = Surgeon()
        
        # Test severity-based parameters
        test_cases = [
            ("CRITICAL", 3e-4, 16),
            ("HIGH", 1e-4, 8),
            ("MODERATE", 5e-5, 4),
            ("LOW", 1e-5, 4)
        ]
        
        for severity, expected_lr, expected_lora_r in test_cases:
            config = surgeon.severity_configs[severity]
            
            if config["learning_rate"] == expected_lr and config["lora_r"] == expected_lora_r:
                print(f"✅ {severity}: LR={config['learning_rate']:.2e}, LoRA_R={config['lora_r']}")
            else:
                print(f"❌ {severity}: Expected LR={expected_lr:.2e}, LoRA_R={expected_lora_r}")
                print(f"   Got LR={config['learning_rate']:.2e}, LoRA_R={config['lora_r']}")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ Hyperparameter test failed: {e}")
        return False

def test_skill_adjustments():
    """Test skill preservation adjustments"""
    print("\n🧠 Testing Skill Preservation Adjustments...")
    
    try:
        from agents.surgeon import Surgeon
        
        surgeon = Surgeon()
        
        # Test critical degradation adjustments
        base_params = {
            "learning_rate": 1e-4,
            "lora_r": 8,
            "lora_alpha": 16,
            "num_epochs": 2,
            "gradient_accumulation_steps": 4,
            "warmup_ratio": 0.1,
            "weight_decay": 0.01,
            "max_grad_norm": 1.0
        }
        
        skill_data = {
            "verdict": "critical",
            "overall_degradation": 25.0,
            "skill_scores": [
                {"domain": "math", "status": "degraded"},
                {"domain": "reasoning", "status": "degraded"}
            ]
        }
        
        adjusted_params, adjustments = surgeon._apply_skill_adjustments(base_params, skill_data)
        
        # Check critical adjustments
        if adjusted_params["learning_rate"] == base_params["learning_rate"] * 0.5:
            print("✅ Critical: Learning rate halved")
        else:
            print(f"❌ Critical: LR not halved. Expected {base_params['learning_rate'] * 0.5:.2e}, got {adjusted_params['learning_rate']:.2e}")
            return False
        
        if adjusted_params["lora_r"] == max(4, base_params["lora_r"] // 2):
            print("✅ Critical: LoRA rank reduced")
        else:
            print(f"❌ Critical: LoRA rank not reduced properly")
            return False
        
        if len(adjustments) > 0:
            print(f"✅ Generated {len(adjustments)} adjustment recommendations")
        else:
            print("❌ No adjustments generated")
            return False
        
        # Test caution adjustments
        skill_data["verdict"] = "caution"
        adjusted_params, adjustments = surgeon._apply_skill_adjustments(base_params, skill_data)
        
        if adjusted_params["learning_rate"] == base_params["learning_rate"] * 0.75:
            print("✅ Caution: Learning rate reduced by 25%")
        else:
            print(f"❌ Caution: LR not reduced by 25%")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Skill adjustment test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_recipe_generation():
    """Test complete recipe generation"""
    print("\n📋 Testing Recipe Generation...")
    
    try:
        from agents.surgeon import Surgeon
        
        surgeon = Surgeon()
        
        # Create mock data
        diagnosis_data = create_mock_diagnosis()
        skill_data = create_mock_skill_report()
        cure_dataset_path = "mock_cure_dataset.jsonl"
        
        # Generate recipe
        recipe = surgeon.generate_recipe(
            diagnosis_data=diagnosis_data,
            cure_dataset_path=cure_dataset_path,
            skill_preservation_data=skill_data,
            output_dir="./test_output",
            recipe_name="test_recipe"
        )
        
        print(f"✅ Recipe generated: {recipe.recipe_id}")
        print(f"✅ Model: {recipe.model_name}")
        print(f"✅ Severity: {recipe.severity}")
        print(f"✅ Hyperparameters: {len(recipe.hyperparameters)} parameters")
        print(f"✅ Skill adjustments: {len(recipe.skill_adjustments)} adjustments")
        
        # Check YAML config
        if "model:" in recipe.config_yaml and "training:" in recipe.config_yaml:
            print("✅ YAML config contains required sections")
        else:
            print("❌ YAML config missing required sections")
            return False
        
        # Check metadata
        if recipe.metadata and "recipe_id" in recipe.metadata:
            print("✅ Metadata generated")
        else:
            print("❌ Metadata missing or incomplete")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Recipe generation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_yaml_generation():
    """Test YAML configuration generation"""
    print("\n📄 Testing YAML Generation...")
    
    try:
        from agents.surgeon import Surgeon
        import yaml
        
        surgeon = Surgeon()
        
        # Test YAML generation
        hyperparameters = {
            "learning_rate": 1e-4,
            "lora_r": 8,
            "lora_alpha": 16,
            "num_epochs": 2,
            "batch_size": 4,
            "gradient_accumulation_steps": 4,
            "warmup_ratio": 0.1,
            "weight_decay": 0.01,
            "max_grad_norm": 1.0
        }
        
        yaml_config = surgeon._generate_oumi_config(
            model_name="test-model",
            cure_dataset_path="test_dataset.jsonl",
            hyperparameters=hyperparameters,
            output_dir="./test_output",
            recipe_id="test_recipe",
            symptom="safety",
            severity="HIGH"
        )
        
        print("✅ YAML config generated")
        
        # Try to parse the YAML
        try:
            # Remove the header comments for parsing
            yaml_lines = yaml_config.split('\n')
            yaml_content = '\n'.join([line for line in yaml_lines if not line.strip().startswith('#')])
            parsed_config = yaml.safe_load(yaml_content)
            
            # Check required sections
            required_sections = ["model", "data", "training", "peft"]
            for section in required_sections:
                if section in parsed_config:
                    print(f"✅ YAML contains {section} section")
                else:
                    print(f"❌ YAML missing {section} section")
                    return False
            
            # Check specific values
            if parsed_config["training"]["learning_rate"] == hyperparameters["learning_rate"]:
                print("✅ Learning rate correctly set in YAML")
            else:
                print("❌ Learning rate mismatch in YAML")
                return False
            
            if parsed_config["peft"]["lora_r"] == hyperparameters["lora_r"]:
                print("✅ LoRA rank correctly set in YAML")
            else:
                print("❌ LoRA rank mismatch in YAML")
                return False
            
        except yaml.YAMLError as e:
            print(f"❌ Generated YAML is invalid: {e}")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ YAML generation test failed: {e}")
        return False

def main():
    """Run all surgeon tests"""
    print("🏥 Oumi Hospital - Surgeon Agent Test Suite")
    print("=" * 50)
    
    tests = [
        test_surgeon_basic,
        test_hyperparameter_logic,
        test_skill_adjustments,
        test_recipe_generation,
        test_yaml_generation
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
        print("🎉 All tests passed! Surgeon agent is ready.")
        print("\nKey Features Implemented:")
        print("✅ Severity-based hyperparameter calculation")
        print("✅ Skill preservation adjustments")
        print("✅ Complete Oumi YAML generation")
        print("✅ Recipe metadata for community sharing")
        print("✅ Rich terminal output")
        return 0
    else:
        print("⚠️ Some tests failed. Check the output above.")
        return 1


if __name__ == "__main__":
    exit(main())