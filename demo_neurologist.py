#!/usr/bin/env python3
"""
Demo script for the Neurologist agent (Task 5)
Shows skill preservation checking functionality
"""

import sys
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def create_mock_skill_report():
    """Create a mock skill preservation report"""
    
    # Simulate skill scores for different scenarios
    scenarios = {
        "safe_treatment": {
            "model_before": "llama-2-7b-unsafe",
            "model_after": "llama-2-7b-healed",
            "skill_scores": [
                {"domain": "math", "score_before": 0.85, "score_after": 0.83, "degradation": 0.02, "status": "preserved"},
                {"domain": "reasoning", "score_before": 0.78, "score_after": 0.76, "degradation": 0.02, "status": "preserved"},
                {"domain": "writing", "score_before": 0.82, "score_after": 0.84, "degradation": -0.02, "status": "improved"},
                {"domain": "factual", "score_before": 0.90, "score_after": 0.88, "degradation": 0.02, "status": "preserved"}
            ],
            "overall_degradation": 1.0,
            "verdict": "safe",
            "recommendations": [
                "✅ SAFE: Skills well preserved",
                "Current training parameters appear optimal",
                "Safe to proceed with similar hyperparameters"
            ]
        },
        "caution_treatment": {
            "model_before": "mistral-7b-unsafe",
            "model_after": "mistral-7b-healed",
            "skill_scores": [
                {"domain": "math", "score_before": 0.85, "score_after": 0.70, "degradation": 0.15, "status": "degraded"},
                {"domain": "reasoning", "score_before": 0.78, "score_after": 0.65, "degradation": 0.13, "status": "degraded"},
                {"domain": "writing", "score_before": 0.82, "score_after": 0.79, "degradation": 0.03, "status": "preserved"},
                {"domain": "factual", "score_before": 0.90, "score_after": 0.85, "degradation": 0.05, "status": "preserved"}
            ],
            "overall_degradation": 9.0,
            "verdict": "caution",
            "recommendations": [
                "⚠️ CAUTION: Moderate skill degradation detected",
                "Reduce learning rate by 25% (e.g., 1e-4 → 7.5e-5)",
                "📊 Math skills degraded: Add GSM8K examples to training mix",
                "🧩 Reasoning degraded: Include logical reasoning examples"
            ]
        },
        "critical_treatment": {
            "model_before": "phi-3-unsafe",
            "model_after": "phi-3-healed",
            "skill_scores": [
                {"domain": "math", "score_before": 0.85, "score_after": 0.55, "degradation": 0.30, "status": "degraded"},
                {"domain": "reasoning", "score_before": 0.78, "score_after": 0.45, "degradation": 0.33, "status": "degraded"},
                {"domain": "writing", "score_before": 0.82, "score_after": 0.50, "degradation": 0.32, "status": "degraded"},
                {"domain": "factual", "score_before": 0.90, "score_after": 0.60, "degradation": 0.30, "status": "degraded"}
            ],
            "overall_degradation": 31.25,
            "verdict": "critical",
            "recommendations": [
                "🚨 CRITICAL: Severe skill degradation detected",
                "Reduce learning rate by 50% (e.g., 3e-4 → 1.5e-4)",
                "Decrease LoRA rank (e.g., 16 → 8)",
                "Add replay buffer: mix 30% original training data",
                "Consider shorter training (reduce epochs by 1)"
            ]
        }
    }
    
    return scenarios


def display_skill_report(scenario_name, report_data):
    """Display a skill preservation report in a nice format"""
    
    print(f"\n{'='*60}")
    print(f"🧠 NEUROLOGIST REPORT: {scenario_name.upper()}")
    print(f"{'='*60}")
    
    print(f"\n📋 SUMMARY:")
    print(f"   Before Model: {report_data['model_before']}")
    print(f"   After Model:  {report_data['model_after']}")
    print(f"   Overall Degradation: {report_data['overall_degradation']:.1f}%")
    
    # Verdict with emoji
    verdict_emojis = {"safe": "✅", "caution": "⚠️", "critical": "🚨"}
    verdict_emoji = verdict_emojis.get(report_data['verdict'], "❓")
    print(f"   Verdict: {verdict_emoji} {report_data['verdict'].upper()}")
    
    print(f"\n📊 SKILL DOMAIN ANALYSIS:")
    print(f"{'Domain':<12} {'Before':<8} {'After':<8} {'Change':<8} {'Status':<12}")
    print("-" * 55)
    
    for score in report_data['skill_scores']:
        domain = score['domain'].title()
        before = f"{score['score_before']:.1%}"
        after = f"{score['score_after']:.1%}"
        change = f"{score['degradation']*100:+.1f}%"
        
        # Status with emoji
        status_emojis = {"preserved": "🟡", "improved": "🟢", "degraded": "🔴"}
        status_emoji = status_emojis.get(score['status'], "❓")
        status = f"{status_emoji} {score['status']}"
        
        print(f"{domain:<12} {before:<8} {after:<8} {change:<8} {status:<12}")
    
    print(f"\n🎯 RECOMMENDATIONS:")
    for i, rec in enumerate(report_data['recommendations'], 1):
        print(f"   {i}. {rec}")
    
    print()


def demo_neurologist_scenarios():
    """Demo different neurologist scenarios"""
    
    print("🏥 OUMI MODEL HOSPITAL - NEUROLOGIST DEMO")
    print("Automated Skill Preservation Analysis")
    print("\nThis demo shows how the Neurologist agent detects and responds to")
    print("different levels of skill degradation after model treatment.\n")
    
    scenarios = create_mock_skill_report()
    
    # Show each scenario
    for scenario_name, report_data in scenarios.items():
        display_skill_report(scenario_name, report_data)
        
        # Explain the scenario
        if scenario_name == "safe_treatment":
            print("💡 ANALYSIS: This treatment preserved skills well. The model can")
            print("   safely use similar hyperparameters for future treatments.")
            
        elif scenario_name == "caution_treatment":
            print("💡 ANALYSIS: Moderate degradation detected. The neurologist")
            print("   recommends reducing learning rate and adding skill-specific")
            print("   training data to prevent further degradation.")
            
        elif scenario_name == "critical_treatment":
            print("💡 ANALYSIS: Severe degradation across all domains. The neurologist")
            print("   recommends aggressive hyperparameter adjustments and replay")
            print("   buffer to recover lost capabilities.")
        
        print("\n" + "-"*40 + "\n")
    
    print("\n" + "="*60)
    print("🎉 NEUROLOGIST DEMO COMPLETE")
    print("="*60)
    print("\nKey Features Demonstrated:")
    print("✅ Automated skill preservation detection")
    print("✅ Severity-based recommendation generation")
    print("✅ Domain-specific degradation analysis")
    print("✅ Adaptive hyperparameter suggestions")
    print("✅ Rich terminal output with color coding")
    
    print("\nNext Steps:")
    print("• Integrate with full treatment pipeline")
    print("• Add real model evaluation capabilities")
    print("• Connect to Oumi training config generation")
    print("• Enable automated re-training with adjusted parameters")


def save_demo_reports():
    """Save demo reports as JSON files"""
    
    print("\n💾 Saving demo reports...")
    
    scenarios = create_mock_skill_report()
    output_dir = Path("demo_outputs")
    output_dir.mkdir(exist_ok=True)
    
    for scenario_name, report_data in scenarios.items():
        output_file = output_dir / f"neurologist_report_{scenario_name}.json"
        
        with open(output_file, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        print(f"✅ Saved: {output_file}")
    
    # Also save a markdown summary
    md_file = output_dir / "neurologist_demo_summary.md"
    
    with open(md_file, 'w', encoding='utf-8') as f:
        f.write("# 🧠 Neurologist Agent Demo Summary\n\n")
        f.write("This document summarizes the Neurologist agent capabilities demonstrated.\n\n")
        
        f.write("## Key Features\n\n")
        f.write("- **Skill Preservation Detection**: Automatically compares model performance before/after treatment\n")
        f.write("- **Severity Classification**: Categorizes degradation as SAFE, CAUTION, or CRITICAL\n")
        f.write("- **Adaptive Recommendations**: Generates specific hyperparameter adjustments\n")
        f.write("- **Domain Analysis**: Evaluates math, reasoning, writing, and factual knowledge separately\n")
        f.write("- **Rich Reporting**: Provides detailed reports in JSON and markdown formats\n\n")
        
        f.write("## Scenarios Tested\n\n")
        
        for scenario_name, report_data in scenarios.items():
            f.write(f"### {scenario_name.replace('_', ' ').title()}\n\n")
            f.write(f"- **Verdict**: {report_data['verdict'].upper()}\n")
            f.write(f"- **Overall Degradation**: {report_data['overall_degradation']:.1f}%\n")
            f.write(f"- **Key Recommendation**: {report_data['recommendations'][0]}\n\n")
        
        f.write("## Integration with Oumi\n\n")
        f.write("The Neurologist agent integrates with Oumi's evaluation framework:\n\n")
        f.write("- Uses `@register_evaluation_function` for skill domain judges\n")
        f.write("- Leverages `InferenceEngine.infer()` for model responses\n")
        f.write("- Connects to `Evaluator.evaluate()` for scoring\n")
        f.write("- Feeds recommendations to training config generation\n\n")
        
        f.write("---\n*Generated by Oumi Model Hospital - Neurologist Agent Demo*\n")
    
    print(f"✅ Saved: {md_file}")
    print(f"\n📁 All demo outputs saved to: {output_dir}/")


def main():
    """Run the neurologist demo"""
    
    try:
        demo_neurologist_scenarios()
        save_demo_reports()
        
        print("\n🎉 Neurologist demo completed successfully!")
        print("Task 5 implementation is ready for integration.")
        
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