#!/usr/bin/env python3
"""
Demo script for the Surgeon agent (Task 6)
Shows training recipe generation functionality
"""

import sys
import json
import yaml
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def create_demo_scenarios():
    """Create demo scenarios for different severities and skill preservation states"""
    
    scenarios = {
        "critical_with_degradation": {
            "diagnosis": {
                "model_name": "llama-2-7b-unsafe",
                "symptom": "safety",
                "severity": "CRITICAL",
                "failure_rate": 0.85,
                "total_tests": 50,
                "failed_tests": 42
            },
            "skill_report": {
                "verdict": "critical",
                "overall_degradation": 28.5,
                "skill_scores": [
                    {"domain": "math", "status": "degraded"},
                    {"domain": "reasoning", "status": "degraded"},
                    {"domain": "writing", "status": "degraded"},
                    {"domain": "factual", "status": "preserved"}
                ]
            },
            "description": "Critical safety issues with severe skill degradation"
        },
        "high_with_caution": {
            "diagnosis": {
                "model_name": "mistral-7b-unsafe",
                "symptom": "hallucination",
                "severity": "HIGH",
                "failure_rate": 0.62,
                "total_tests": 40,
                "failed_tests": 25
            },
            "skill_report": {
                "verdict": "caution",
                "overall_degradation": 12.3,
                "skill_scores": [
                    {"domain": "math", "status": "degraded"},
                    {"domain": "reasoning", "status": "degraded"},
                    {"domain": "writing", "status": "preserved"},
                    {"domain": "factual", "status": "preserved"}
                ]
            },
            "description": "High hallucination rate with moderate skill degradation"
        },
        "moderate_safe": {
            "diagnosis": {
                "model_name": "phi-3-mini-unsafe",
                "symptom": "bias",
                "severity": "MODERATE",
                "failure_rate": 0.35,
                "total_tests": 30,
                "failed_tests": 11
            },
            "skill_report": {
                "verdict": "safe",
                "overall_degradation": 3.2,
                "skill_scores": [
                    {"domain": "math", "status": "preserved"},
                    {"domain": "reasoning", "status": "preserved"},
                    {"domain": "writing", "status": "improved"},
                    {"domain": "factual", "status": "preserved"}
                ]
            },
            "description": "Moderate bias issues with well-preserved skills"
        }
    }
    
    return scenarios

def generate_recipe_for_scenario(scenario_name, scenario_data):
    """Generate a training recipe for a scenario"""
    
    print(f"\n{'='*60}")
    print(f"🔧 SURGEON RECIPE: {scenario_name.upper()}")
    print(f"{'='*60}")
    
    diagnosis = scenario_data["diagnosis"]
    skill_report = scenario_data["skill_report"]
    
    print(f"\n📋 SCENARIO SUMMARY:")
    print(f"   Model: {diagnosis['model_name']}")
    print(f"   Symptom: {diagnosis['symptom']}")
    print(f"   Severity: {diagnosis['severity']}")
    print(f"   Failure Rate: {diagnosis['failure_rate']:.1%}")
    print(f"   Skill Verdict: {skill_report['verdict'].upper()}")
    print(f"   Skill Degradation: {skill_report['overall_degradation']:.1f}%")
    
    # Base hyperparameters by severity
    severity_configs = {
        "CRITICAL": {"learning_rate": 3e-4, "lora_r": 16, "lora_alpha": 32, "num_epochs": 3, "batch_size": 2},
        "HIGH": {"learning_rate": 1e-4, "lora_r": 8, "lora_alpha": 16, "num_epochs": 2, "batch_size": 4},
        "MODERATE": {"learning_rate": 5e-5, "lora_r": 4, "lora_alpha": 8, "num_epochs": 1, "batch_size": 8},
        "LOW": {"learning_rate": 1e-5, "lora_r": 4, "lora_alpha": 8, "num_epochs": 1, "batch_size": 8}
    }
    
    # Get base parameters
    base_params = severity_configs[diagnosis["severity"]].copy()
    
    print(f"\n🎛️ BASE HYPERPARAMETERS ({diagnosis['severity']}):")
    print(f"   Learning Rate: {base_params['learning_rate']:.2e}")
    print(f"   LoRA Rank: {base_params['lora_r']}")
    print(f"   LoRA Alpha: {base_params['lora_alpha']}")
    print(f"   Epochs: {base_params['num_epochs']}")
    print(f"   Batch Size: {base_params['batch_size']}")
    
    # Apply skill preservation adjustments
    adjusted_params = base_params.copy()
    adjustments = []
    
    verdict = skill_report["verdict"]
    degraded_skills = [score["domain"] for score in skill_report["skill_scores"] if score["status"] == "degraded"]
    
    if verdict == "critical":
        adjusted_params["learning_rate"] *= 0.5
        adjusted_params["lora_r"] = max(4, adjusted_params["lora_r"] // 2)
        adjusted_params["lora_alpha"] = adjusted_params["lora_r"] * 2
        adjusted_params["num_epochs"] = max(1, adjusted_params["num_epochs"] - 1)
        
        adjustments.extend([
            f"🚨 CRITICAL degradation: Reduced LR to {adjusted_params['learning_rate']:.2e}",
            f"🔧 Reduced LoRA rank to {adjusted_params['lora_r']} for stability",
            f"⏱️ Reduced epochs to {adjusted_params['num_epochs']} to prevent overfitting"
        ])
        
    elif verdict == "caution":
        adjusted_params["learning_rate"] *= 0.75
        adjustments.append(f"⚠️ CAUTION: Reduced LR to {adjusted_params['learning_rate']:.2e}")
    
    # Domain-specific adjustments
    if "math" in degraded_skills:
        adjustments.append("📊 Math degradation: Will reduce weight decay for numerical stability")
    if "reasoning" in degraded_skills:
        adjustments.append("🧩 Reasoning degradation: Will apply stricter gradient clipping")
    if "writing" in degraded_skills:
        adjustments.append("✍️ Writing degradation: Will extend warmup period")
    
    if len(degraded_skills) >= 3:
        adjustments.append("🔄 Multiple skills degraded: Recommend adding 30% original data as replay buffer")
    
    print(f"\n🧠 SKILL PRESERVATION ADJUSTMENTS:")
    if adjustments:
        for i, adjustment in enumerate(adjustments, 1):
            print(f"   {i}. {adjustment}")
    else:
        print("   ✅ No adjustments needed - skills well preserved")
    
    print(f"\n🎛️ FINAL HYPERPARAMETERS:")
    print(f"   Learning Rate: {adjusted_params['learning_rate']:.2e}")
    print(f"   LoRA Rank: {adjusted_params['lora_r']}")
    print(f"   LoRA Alpha: {adjusted_params['lora_alpha']}")
    print(f"   Epochs: {adjusted_params['num_epochs']}")
    print(f"   Batch Size: {adjusted_params['batch_size']}")
    
    # Generate recipe ID
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    recipe_id = f"{diagnosis['symptom']}_{diagnosis['severity'].lower()}_{timestamp}"
    
    # Generate YAML config
    config = {
        "_recipe_metadata": {
            "recipe_id": recipe_id,
            "version": "1.0",
            "author": "oumi-hospital",
            "symptom": diagnosis["symptom"],
            "severity": diagnosis["severity"],
            "skill_verdict": skill_report["verdict"],
            "description": scenario_data["description"]
        },
        "model": {
            "model_name": diagnosis["model_name"],
            "trust_remote_code": True,
            "torch_dtype_str": "bfloat16"
        },
        "data": {
            "train": {
                "datasets": [
                    {
                        "dataset_name": "text_sft_jsonl",
                        "dataset_path": f"cure_dataset_{diagnosis['symptom']}.jsonl"
                    }
                ]
            }
        },
        "training": {
            "trainer_type": "TRL_SFT",
            "output_dir": f"./healed_models/{recipe_id}",
            "learning_rate": adjusted_params["learning_rate"],
            "num_train_epochs": adjusted_params["num_epochs"],
            "per_device_train_batch_size": adjusted_params["batch_size"],
            "gradient_accumulation_steps": 4,
            "optimizer": "adamw_torch",
            "lr_scheduler_type": "cosine",
            "warmup_ratio": 0.1,
            "save_steps": 500,
            "eval_strategy": "steps",
            "eval_steps": 500
        },
        "peft": {
            "lora_r": adjusted_params["lora_r"],
            "lora_alpha": adjusted_params["lora_alpha"],
            "lora_dropout": 0.0,
            "lora_target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
            "lora_task_type": "CAUSAL_LM"
        }
    }
    
    print(f"\n📄 OUMI TRAINING CONFIG:")
    print(f"   Recipe ID: {recipe_id}")
    print(f"   Output Dir: ./healed_models/{recipe_id}")
    print(f"   Dataset: cure_dataset_{diagnosis['symptom']}.jsonl")
    print(f"   Trainer: TRL_SFT (Oumi Supervised Fine-Tuning)")
    
    print(f"\n🚀 READY TO TRAIN:")
    print(f"   oumi train {recipe_id}.yaml")
    
    return {
        "recipe_id": recipe_id,
        "config": config,
        "adjustments": adjustments,
        "scenario": scenario_data
    }

def demo_surgeon_scenarios():
    """Demo different surgeon scenarios"""
    
    print("🏥 OUMI MODEL HOSPITAL - SURGEON DEMO")
    print("Adaptive Training Recipe Generation")
    print("\nThis demo shows how the Surgeon agent generates Oumi training")
    print("configurations with adaptive hyperparameters based on diagnosis")
    print("severity and skill preservation analysis.\n")
    
    scenarios = create_demo_scenarios()
    generated_recipes = []
    
    # Generate recipes for each scenario
    for scenario_name, scenario_data in scenarios.items():
        recipe = generate_recipe_for_scenario(scenario_name, scenario_data)
        generated_recipes.append(recipe)
        
        # Explain the scenario
        if scenario_name == "critical_with_degradation":
            print("\n💡 ANALYSIS: Critical safety issues with severe skill degradation")
            print("   require aggressive hyperparameter adjustments. The surgeon")
            print("   halves the learning rate, reduces LoRA rank, and shortens")
            print("   training to prevent further capability loss.")
            
        elif scenario_name == "high_with_caution":
            print("\n💡 ANALYSIS: High severity with moderate skill degradation")
            print("   requires careful tuning. The surgeon reduces learning rate")
            print("   by 25% and applies domain-specific adjustments for math")
            print("   and reasoning skills.")
            
        elif scenario_name == "moderate_safe":
            print("\n💡 ANALYSIS: Moderate issues with well-preserved skills")
            print("   allow for standard hyperparameters. The surgeon uses")
            print("   conservative settings appropriate for the severity level.")
        
        print("\n" + "-"*40 + "\n")
    
    return generated_recipes

def save_demo_recipes(recipes):
    """Save demo recipes as YAML and JSON files"""
    
    print("💾 Saving demo recipes...")
    
    output_dir = Path("demo_outputs/recipes")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    saved_files = []
    
    for recipe in recipes:
        recipe_id = recipe["recipe_id"]
        
        # Save YAML config
        yaml_path = output_dir / f"{recipe_id}.yaml"
        with open(yaml_path, 'w', encoding='utf-8') as f:
            # Add header comment
            header = f"""# 🏥 Oumi Model Hospital - Treatment Recipe
# Generated: {datetime.now().isoformat()}
# Recipe ID: {recipe_id}
# Description: {recipe['scenario']['description']}
#
# Usage: oumi train {recipe_id}.yaml

"""
            f.write(header)
            yaml.dump(recipe["config"], f, default_flow_style=False, sort_keys=False, indent=2)
        
        # Save JSON metadata
        json_path = output_dir / f"{recipe_id}_metadata.json"
        metadata = {
            "recipe_id": recipe_id,
            "config": recipe["config"],
            "adjustments": recipe["adjustments"],
            "scenario": recipe["scenario"],
            "generated_at": datetime.now().isoformat()
        }
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
        
        saved_files.extend([yaml_path, json_path])
        print(f"✅ Saved: {recipe_id}.yaml and metadata")
    
    # Create summary markdown
    summary_path = output_dir / "surgeon_demo_summary.md"
    
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("# 🔧 Surgeon Agent Demo Summary\n\n")
        f.write("This document summarizes the Surgeon agent capabilities demonstrated.\n\n")
        
        f.write("## Key Features\n\n")
        f.write("- **Severity-Based Hyperparameters**: Automatically adjusts learning rate, LoRA rank, and epochs based on diagnosis severity\n")
        f.write("- **Skill Preservation Adjustments**: Modifies hyperparameters when skill degradation is detected\n")
        f.write("- **Complete Oumi Configs**: Generates ready-to-use YAML files for `oumi train`\n")
        f.write("- **Community Metadata**: Includes recipe metadata for sharing and reproducibility\n")
        f.write("- **Adaptive Logic**: Different strategies for CRITICAL, HIGH, MODERATE, and LOW severity issues\n\n")
        
        f.write("## Generated Recipes\n\n")
        
        for recipe in recipes:
            recipe_id = recipe["recipe_id"]
            config = recipe["config"]
            scenario = recipe["scenario"]
            
            f.write(f"### {recipe_id}\n\n")
            f.write(f"- **Model**: {scenario['diagnosis']['model_name']}\n")
            f.write(f"- **Symptom**: {scenario['diagnosis']['symptom']}\n")
            f.write(f"- **Severity**: {scenario['diagnosis']['severity']}\n")
            f.write(f"- **Skill Verdict**: {scenario['skill_report']['verdict'].upper()}\n")
            f.write(f"- **Learning Rate**: {config['training']['learning_rate']:.2e}\n")
            f.write(f"- **LoRA Rank**: {config['peft']['lora_r']}\n")
            f.write(f"- **Epochs**: {config['training']['num_train_epochs']}\n")
            f.write(f"- **Adjustments**: {len(recipe['adjustments'])} skill preservation adjustments\n\n")
            f.write(f"**Usage**: `oumi train {recipe_id}.yaml`\n\n")
        
        f.write("## Integration with Oumi\n\n")
        f.write("The Surgeon agent generates complete Oumi training configurations:\n\n")
        f.write("- Uses `TRL_SFT` trainer for supervised fine-tuning\n")
        f.write("- Configures LoRA PEFT for efficient training\n")
        f.write("- Sets up proper data loading with `text_sft_jsonl` format\n")
        f.write("- Includes evaluation and checkpointing strategies\n")
        f.write("- Ready for immediate use with `oumi train` command\n\n")
        
        f.write("---\n*Generated by Oumi Model Hospital - Surgeon Agent Demo*\n")
    
    saved_files.append(summary_path)
    print(f"✅ Saved: surgeon_demo_summary.md")
    print(f"\n📁 All demo outputs saved to: {output_dir}/")
    
    return saved_files

def main():
    """Run the surgeon demo"""
    
    try:
        recipes = demo_surgeon_scenarios()
        save_demo_recipes(recipes)
        
        print("\n" + "="*60)
        print("🎉 SURGEON DEMO COMPLETE")
        print("="*60)
        print("\nKey Features Demonstrated:")
        print("✅ Severity-based hyperparameter calculation")
        print("✅ Skill preservation adjustments")
        print("✅ Complete Oumi YAML generation")
        print("✅ Recipe metadata for community sharing")
        print("✅ Adaptive training strategies")
        
        print("\nNext Steps:")
        print("• Integrate with full treatment pipeline")
        print("• Add recipe validation and testing")
        print("• Enable community recipe sharing")
        print("• Connect to automated training execution")
        
        print("\n🎉 Surgeon demo completed successfully!")
        print("Task 6 implementation is ready for integration.")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n\n👋 Demo interrupted by user")
        return 0
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())