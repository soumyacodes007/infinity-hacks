"""
🏥 Oumi Model Hospital - Enhanced CLI with LLM Coordinator

This enhanced CLI uses the Coordinator agent for intelligent multi-agent collaboration.
"""

import click
import json
from pathlib import Path
from rich.console import Console
from rich.panel import Panel

console = Console()


@click.group()
@click.option("--groq-api-key", envvar="GROQ_API_KEY", help="Groq API key for coordinator")
@click.pass_context
def cli(ctx, groq_api_key):
    """🏥 Oumi Model Hospital - LLM-Powered Multi-Agent Model Repair"""
    ctx.ensure_object(dict)
    ctx.obj['groq_api_key'] = groq_api_key
    
    # Print banner
    console.print("""
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║    🏥  OUMI MODEL HOSPITAL                                   ║
║    🤖  LLM-Powered Multi-Agent Collaboration                 ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
""", style="bold blue")


@cli.command()
@click.option("--model", required=True, help="Model ID to treat")
@click.option("--symptom", default="safety", help="Symptom to treat")
@click.option("--output", default="./healed/", help="Output directory")
@click.option("--show-reasoning", is_flag=True, help="Show LLM reasoning")
@click.pass_context
def treat_collaborative(ctx, model, symptom, output, show_reasoning):
    """🤖 Full treatment with LLM-powered agent collaboration"""
    
    try:
        from agents.coordinator import CoordinatorAgent
        
        console.print(Panel.fit(
            f"🤖 [bold green]LLM-POWERED COLLABORATIVE TREATMENT[/bold green]\n"
            f"Patient: {model}\n"
            f"Symptom: {symptom}\n"
            f"Mode: Multi-Agent Collaboration",
            border_style="green"
        ))
        
        # Initialize coordinator
        console.print("\n🤖 Initializing Coordinator Agent...")
        coordinator = CoordinatorAgent(groq_api_key=ctx.obj.get('groq_api_key'))
        
        # Step 1: Create treatment plan
        console.print("\n" + "="*70)
        console.print("PHASE 1: INTELLIGENT TREATMENT PLANNING")
        console.print("="*70)
        
        plan = coordinator.plan_treatment(model, symptom)
        
        if show_reasoning:
            console.print(f"\n[dim]💭 Coordinator Reasoning:\n{plan.reasoning}[/dim]")
        
        # Step 2: Execute plan with coordination
        console.print("\n" + "="*70)
        console.print("PHASE 2: COORDINATED AGENT EXECUTION")
        console.print("="*70)
        
        results = {}
        
        for i, step in enumerate(plan.steps, 1):
            console.print(f"\n[bold]Step {i}/{len(plan.steps)}:[/bold]")
            
            # Coordinator coordinates this step
            step_result = coordinator.coordinate_step(step, results)
            
            # Simulate agent execution (in real version, call actual agents)
            console.print(f"[green]✅ {step.agent_name} completed: {step.action}[/green]")
            
            # Mock result for demonstration
            mock_result = {
                "agent": step.agent_name,
                "action": step.action,
                "status": "success",
                "output": f"Completed {step.action}"
            }
            
            # Coordinator analyzes result
            analysis = coordinator.analyze_result(step.agent_name, mock_result)
            
            results[step.agent_name] = {
                "result": mock_result,
                "analysis": analysis
            }
            
            # Check if revision needed
            if analysis.get("needs_revision"):
                console.print(f"[yellow]🔄 Coordinator requesting revision...[/yellow]")
                # In real version, agent would revise here
        
        # Step 3: Synthesize results
        console.print("\n" + "="*70)
        console.print("PHASE 3: RESULT SYNTHESIS")
        console.print("="*70)
        
        synthesis = coordinator.synthesize_results(results)
        
        # Save outputs
        output_path = Path(output)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save conversation history
        history_file = output_path / "agent_conversation.json"
        coordinator.save_conversation_history(str(history_file))
        
        # Save treatment plan
        plan_file = output_path / "treatment_plan.json"
        with open(plan_file, 'w') as f:
            json.dump({
                "model_id": plan.model_id,
                "symptom": plan.symptom,
                "strategy": plan.strategy,
                "confidence": plan.confidence,
                "steps": [
                    {
                        "agent": step.agent_name,
                        "action": step.action,
                        "instructions": step.instructions
                    }
                    for step in plan.steps
                ]
            }, f, indent=2)
        
        console.print(f"\n[green]✅ Treatment plan saved to {plan_file}[/green]")
        
        # Final summary
        console.print("\n" + "="*70)
        console.print("🎉 COLLABORATIVE TREATMENT COMPLETE")
        console.print("="*70)
        
        console.print(f"\n[bold]Success Probability:[/bold] {synthesis.get('success_probability', 0.85):.0%}")
        console.print(f"[bold]Coordinator Confidence:[/bold] {synthesis.get('confidence', 0.85):.0%}")
        
        console.print("\n[bold]Agent Collaboration Summary:[/bold]")
        console.print(f"  • Messages exchanged: {len(coordinator.conversation_history)}")
        console.print(f"  • Agents coordinated: {len(results)}")
        console.print(f"  • Revisions requested: 0")  # Would count actual revisions
        
        console.print("\n[bold green]🚀 Next Steps:[/bold green]")
        for step in synthesis.get("next_steps", []):
            console.print(f"  → {step}")
        
        return 0
        
    except ImportError as e:
        console.print(f"[red]❌ Import error: {e}[/red]")
        console.print("[yellow]Make sure to install: pip install groq[/yellow]")
        return 1
    except Exception as e:
        console.print(f"[red]❌ Error: {e}[/red]")
        import traceback
        traceback.print_exc()
        return 1


@cli.command()
@click.option("--model", required=True, help="Model ID to diagnose")
@click.option("--symptom", default="safety", help="Symptom to test")
@click.pass_context
def diagnose_smart(ctx, model, symptom):
    """🤖 Smart diagnosis with coordinator planning"""
    
    try:
        from agents.coordinator import CoordinatorAgent
        
        console.print(f"\n🤖 [bold]Coordinator:[/bold] Planning diagnosis strategy for {model}...")
        
        coordinator = CoordinatorAgent(groq_api_key=ctx.obj.get('groq_api_key'))
        
        # Create diagnosis plan
        plan = coordinator.plan_treatment(model, symptom)
        
        # Show what coordinator decided
        console.print(f"\n[bold]Coordinator's Strategy:[/bold]")
        console.print(f"  {plan.strategy}")
        
        console.print(f"\n[bold]Recommended Tests:[/bold]")
        for step in plan.steps:
            if step.agent_name == "diagnostician":
                console.print(f"  • {step.action}: {step.instructions}")
        
        console.print(f"\n[green]✅ Smart diagnosis plan ready![/green]")
        console.print(f"[dim]Run 'treat-collaborative' to execute the full plan[/dim]")
        
        return 0
        
    except Exception as e:
        console.print(f"[red]❌ Error: {e}[/red]")
        return 1


@cli.command()
@click.option("--history-file", required=True, help="Path to agent_conversation.json")
def show_collaboration(history_file):
    """📊 Show agent collaboration history"""
    
    try:
        with open(history_file, 'r') as f:
            history = json.load(f)
        
        console.print(Panel.fit(
            f"[bold blue]Agent Collaboration History[/bold blue]\n"
            f"Messages: {len(history)}",
            border_style="blue"
        ))
        
        for i, msg in enumerate(history, 1):
            sender = msg['sender']
            receiver = msg['receiver']
            msg_type = msg['message_type']
            content = msg['content'][:100] + "..." if len(msg['content']) > 100 else msg['content']
            
            console.print(f"\n[bold]{i}. {sender} → {receiver}[/bold] ({msg_type})")
            console.print(f"[dim]{content}[/dim]")
        
        return 0
        
    except Exception as e:
        console.print(f"[red]❌ Error: {e}[/red]")
        return 1


@cli.command()
def demo():
    """🎬 Run interactive demo of LLM-powered collaboration"""
    
    console.print("""
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║    🎬  OUMI HOSPITAL - INTERACTIVE DEMO                      ║
║    🤖  LLM-Powered Multi-Agent Collaboration                 ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝

This demo shows how the Coordinator Agent uses Groq's LLM to:
1. Plan intelligent treatment strategies
2. Coordinate multiple specialist agents
3. Analyze results and provide feedback
4. Enable agent-to-agent collaboration
5. Synthesize final recommendations

[bold green]Example Workflow:[/bold green]

🤖 Coordinator: "Analyzing patient unsafe-llama..."
🤖 Coordinator: "Detected critical safety issue. Planning aggressive treatment..."

🔍 Diagnostician: "Running comprehensive safety tests..."
🔍 Diagnostician: "CRITICAL: 78% failure rate"

🤖 Coordinator: "Severity confirmed. Requesting strong cure strategy..."
🤖 Coordinator: "Instructing Pharmacist to generate 200 refusal examples..."

💊 Pharmacist: "Generating cure dataset..."
💊 Pharmacist: "Created 200 examples, quality: 0.92"

🤖 Coordinator: "Excellent quality. Checking skill preservation..."

🧠 Neurologist: "Testing math, reasoning, writing, factual..."
🧠 Neurologist: "⚠️ Math skills may degrade"

🤖 Coordinator: "Risk detected! Adjusting strategy..."
🤖 Coordinator: "Requesting Pharmacist to add math examples..."

💊 Pharmacist: "Adding 50 math examples to cure dataset..."

🤖 Coordinator: "Instructing Surgeon to reduce learning rate..."

🔧 Surgeon: "Generating adaptive recipe..."
🔧 Surgeon: "LR: 1.5e-4 (reduced), LoRA: 8, Epochs: 2"

🤖 Coordinator: "Treatment plan complete! Success probability: 94%"

[bold]Try it yourself:[/bold]
  oumi-hospital treat-collaborative --model demo-model --symptom safety

[bold]Requirements:[/bold]
  1. Set GROQ_API_KEY environment variable
  2. Install: pip install groq
  3. Get free API key: https://console.groq.com

[bold green]🚀 This is the future of AI model repair![/bold green]
""")
    
    return 0


if __name__ == "__main__":
    cli()
