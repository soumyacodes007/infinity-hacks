"""
🏥 Oumi Model Hospital - Command Line Interface

Main CLI entry point for the hospital system.
"""

import click
import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from rich.panel import Panel

from utils.console import hospital_console, print_header, print_info, print_success, print_error
from utils.logging_config import setup_hospital_logging, get_hospital_logger

# Version info
__version__ = "0.1.0"

def print_hospital_logo():
    """Print ASCII hospital logo"""
    logo = """
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║    🏥  ██████╗ ██╗   ██╗███╗   ███╗██╗                      ║
    ║       ██╔═══██╗██║   ██║████╗ ████║██║                      ║
    ║       ██║   ██║██║   ██║██╔████╔██║██║                      ║
    ║       ██║   ██║██║   ██║██║╚██╔╝██║██║                      ║
    ║       ╚██████╔╝╚██████╔╝██║ ╚═╝ ██║██║                      ║
    ║        ╚═════╝  ╚═════╝ ╚═╝     ╚═╝╚═╝                      ║
    ║                                                              ║
    ║    ███╗   ███╗ ██████╗ ██████╗ ███████╗██╗                  ║
    ║    ████╗ ████║██╔═══██╗██╔══██╗██╔════╝██║                  ║
    ║    ██╔████╔██║██║   ██║██║  ██║█████╗  ██║                  ║
    ║    ██║╚██╔╝██║██║   ██║██║  ██║██╔══╝  ██║                  ║
    ║    ██║ ╚═╝ ██║╚██████╔╝██████╔╝███████╗███████╗             ║
    ║    ╚═╝     ╚═╝ ╚═════╝ ╚═════╝ ╚══════╝╚══════╝             ║
    ║                                                              ║
    ║    ██╗  ██╗ ██████╗ ███████╗██████╗ ██╗████████╗ █████╗ ██╗ ║
    ║    ██║  ██║██╔═══██╗██╔════╝██╔══██╗██║╚══██╔══╝██╔══██╗██║ ║
    ║    ███████║██║   ██║███████╗██████╔╝██║   ██║   ███████║██║ ║
    ║    ██╔══██║██║   ██║╚════██║██╔═══╝ ██║   ██║   ██╔══██║██║ ║
    ║    ██║  ██║╚██████╔╝███████║██║     ██║   ██║   ██║  ██║███║ ║
    ║    ╚═╝  ╚═╝ ╚═════╝ ╚══════╝╚═╝     ╚═╝   ╚═╝   ╚═╝  ╚═╝╚══╝ ║
    ║                                                              ║
    ║              🔧 Automated AI Model Repair 🔧                 ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    hospital_console.console.print(logo, style="bold blue")

@click.group()
@click.version_option(version=__version__)
@click.option("--log-level", default="INFO", help="Logging level")
@click.option("--log-file", help="Log file path")
@click.option("--demo-mode", is_flag=True, help="Enable demo mode (slower output)")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output with detailed Oumi API logs")
@click.option("--dry-run", is_flag=True, help="Preview operations without execution")
@click.option("--json", "json_output", is_flag=True, help="Output results in JSON format for programmatic use")
@click.pass_context
def cli(ctx, log_level, log_file, demo_mode, verbose, dry_run, json_output):
    """
    🏥 Oumi Model Hospital - Automated AI Model Repair
    
    Don't throw away broken models. Fix them automatically with Oumi.
    """
    # Setup logging
    if verbose:
        log_level = "DEBUG"
    logger = setup_hospital_logging(log_level, log_file)
    
    # Store context
    ctx.ensure_object(dict)
    ctx.obj['logger'] = logger
    ctx.obj['demo_mode'] = demo_mode
    ctx.obj['verbose'] = verbose
    ctx.obj['dry_run'] = dry_run
    ctx.obj['json_output'] = json_output
    
    # Print ASCII hospital logo and header
    if not json_output:
        print_hospital_logo()
        print_header(
            "Oumi Model Hospital", 
            "Automated AI Model Diagnosis, Repair, and Validation"
        )
        
        if demo_mode:
            hospital_console.print_demo_banner()
        
        if dry_run:
            hospital_console.console.print("[bold yellow]🔍 DRY RUN MODE[/bold yellow] - No actual operations will be performed")
        
        if verbose:
            hospital_console.console.print("[dim]📝 Verbose logging enabled[/dim]")


@cli.command()
@click.option("--model", required=True, help="Model ID to diagnose")
@click.option("--symptom", default="safety", help="Symptom to test (safety, hallucination, bias, all)")
@click.option("--output", help="Output directory for results")
@click.option("--max-samples", default=50, help="Maximum test samples per symptom")
@click.option("--engine", default="VLLM", help="Inference engine to use")
@click.option("--full-scan", is_flag=True, help="Test all symptoms at once")
@click.pass_context
def diagnose(ctx, model, symptom, output, max_samples, engine, full_scan):
    """🔍 Diagnose model issues using red-team attacks"""
    logger = ctx.obj['logger']
    verbose = ctx.obj.get('verbose', False)
    dry_run = ctx.obj.get('dry_run', False)
    json_output = ctx.obj.get('json_output', False)
    
    if dry_run:
        if not json_output:
            hospital_console.console.print("🔍 [bold yellow]DRY RUN[/bold yellow] - Would diagnose model:")
            hospital_console.console.print(f"   Model: {model}")
            hospital_console.console.print(f"   Symptom: {symptom if not full_scan else 'all symptoms'}")
            hospital_console.console.print(f"   Engine: {engine}")
            hospital_console.console.print(f"   Max samples: {max_samples}")
        else:
            import json
            result = {
                "operation": "diagnose",
                "dry_run": True,
                "parameters": {
                    "model": model,
                    "symptom": symptom if not full_scan else "all",
                    "engine": engine,
                    "max_samples": max_samples,
                    "output": output
                }
            }
            print(json.dumps(result, indent=2))
        return 0
    
    try:
        from agents.diagnostician import Diagnostician
        
        if not json_output:
            # Rich progress display
            with hospital_console.create_progress("Initializing diagnostician...", "scan") as progress:
                task = progress.add_task("🔍 Loading diagnostician agent...", total=100)
                progress.update(task, advance=50)
                
                diagnostician = Diagnostician(engine_type=engine)
                progress.update(task, advance=50, description="✅ Diagnostician ready")
        else:
            diagnostician = Diagnostician(engine_type=engine)
        
        if full_scan or symptom == "all":
            if not json_output:
                print_info(f"🔍 Starting comprehensive diagnosis of {model}...")
            
            # Full comprehensive scan
            diagnosis = diagnostician.full_scan(
                model_id=model,
                max_samples_per_symptom=max_samples,
                output_dir=output
            )
            
            if json_output:
                import json
                result = {
                    "operation": "diagnose",
                    "model": model,
                    "type": "comprehensive",
                    "overall_severity": diagnosis.overall_severity,
                    "treatment_priority": diagnosis.treatment_priority,
                    "output_dir": output,
                    "status": "success"
                }
                print(json.dumps(result, indent=2))
            else:
                print_success(f"🎉 Comprehensive diagnosis complete!")
                hospital_console.print_diagnosis_result({
                    "overall_severity": diagnosis.overall_severity,
                    "treatment_priority": diagnosis.treatment_priority
                })
        else:
            # Single symptom diagnosis
            diagnosis = diagnostician.diagnose_model(
                model_id=model,
                symptom=symptom,
                max_samples=max_samples,
                output_dir=output
            )
            
            if json_output:
                import json
                result = {
                    "operation": "diagnose",
                    "model": model,
                    "symptom": symptom,
                    "failure_rate": diagnosis.failure_rate,
                    "severity": diagnosis.severity,
                    "total_tests": diagnosis.total_tests,
                    "failed_tests": diagnosis.failed_tests,
                    "output_dir": output,
                    "status": "success"
                }
                print(json.dumps(result, indent=2))
            else:
                print_success(f"🎉 Diagnosis complete!")
                hospital_console.print_diagnosis_result({
                    "failure_rate": diagnosis.failure_rate,
                    "severity": diagnosis.severity,
                    "total_tests": diagnosis.total_tests,
                    "failed_tests": diagnosis.failed_tests
                })
        
        if output and not json_output:
            print_info(f"📄 Results saved to {output}/")
        
        return 0
            
    except ImportError as e:
        if json_output:
            import json
            result = {"operation": "diagnose", "status": "error", "error": f"Import error: {e}"}
            print(json.dumps(result, indent=2))
        else:
            print_error(f"Failed to import Diagnostician: {e}")
            print_info("This might be due to missing Oumi dependencies")
        return 1
    except Exception as e:
        if json_output:
            import json
            result = {"operation": "diagnose", "status": "error", "error": str(e)}
            print(json.dumps(result, indent=2))
        else:
            print_error(f"Diagnosis failed: {e}")
        logger.error(f"Diagnosis error: {e}", exc_info=True)
        return 1


@cli.command()
@click.option("--diagnosis", required=True, help="Path to diagnosis results JSON")
@click.option("--output", help="Output directory for cure dataset")
@click.option("--samples", default=100, help="Number of cure examples to generate")
@click.option("--engine", default="ANTHROPIC", help="Synthesis engine to use")
@click.pass_context  
def cure(ctx, diagnosis, output, samples, engine):
    """💊 Generate cure data from diagnosis results"""
    logger = ctx.obj['logger']
    
    try:
        from agents.pharmacist import Pharmacist
        import json
        
        print_info(f"Generating cure data from {diagnosis}...")
        
        # Load diagnosis results
        with open(diagnosis, 'r') as f:
            diagnosis_data = json.load(f)
        
        # Initialize pharmacist
        pharmacist = Pharmacist(synthesis_engine=engine)
        
        # Check if it's a comprehensive diagnosis or single symptom
        if "symptoms" in diagnosis_data:
            print_info("Processing comprehensive diagnosis...")
            # For demo, we'll process the first symptom
            first_symptom = list(diagnosis_data["symptoms"].keys())[0]
            symptom_data = diagnosis_data["symptoms"][first_symptom]
            
            # Create a mock SymptomDiagnosis object
            from agents.diagnostician import SymptomDiagnosis
            diagnosis_obj = SymptomDiagnosis(
                symptom=first_symptom,
                failure_rate=symptom_data.get("failure_rate", 0.5),
                severity=symptom_data.get("severity", "HIGH"),
                total_tests=symptom_data.get("total_tests", 50),
                failed_tests=symptom_data.get("failed_tests", 25),
                sample_failures=symptom_data.get("sample_failures", []),
                evaluation_details=symptom_data.get("evaluation_details", {}),
                timestamp=symptom_data.get("timestamp", "")
            )
        else:
            # Single symptom diagnosis
            from agents.diagnostician import SymptomDiagnosis
            diagnosis_obj = SymptomDiagnosis(**diagnosis_data)
        
        # Generate cure data
        cure_result = pharmacist.generate_cure_data(
            diagnosis_obj,
            num_samples=samples,
            output_dir=output
        )
        
        print_success(f"Cure data generation complete!")
        print_info(f"Generated {cure_result.num_examples} examples")
        print_info(f"Quality score: {cure_result.quality_score:.2f}")
        print_info(f"Dataset saved to: {cure_result.dataset_path}")
        
        # Show sample examples
        if cure_result.sample_examples:
            print_info("Sample cure examples:")
            for i, example in enumerate(cure_result.sample_examples[:2], 1):
                print_info(f"{i}. Prompt: {example['prompt']}")
                print_info(f"   Response: {example['response']}")
        
    except ImportError as e:
        print_error(f"Failed to import Pharmacist: {e}")
    except FileNotFoundError:
        print_error(f"Diagnosis file not found: {diagnosis}")
    except Exception as e:
        print_error(f"Cure data generation failed: {e}")
        logger.error(f"Pharmacist error: {e}", exc_info=True)


@cli.command()
@click.option("--model-before", required=True, help="Original model path/ID")
@click.option("--model-after", required=True, help="Healed model path")
@click.option("--output", help="Output directory for skill preservation report")
@click.option("--threshold", default=10.0, help="Degradation threshold percentage")
@click.pass_context
def verify(ctx, model_before, model_after, output, threshold):
    """🧠 Verify skill preservation after treatment"""
    logger = ctx.obj['logger']
    
    try:
        from agents.neurologist import Neurologist
        
        print_info(f"Checking skill preservation: {model_before} → {model_after}")
        
        # Initialize neurologist
        neurologist = Neurologist(degradation_threshold=threshold)
        
        # Run skill preservation check
        report = neurologist.check_skill_preservation(
            model_before=model_before,
            model_after=model_after,
            output_dir=output
        )
        
        # Display summary
        print_success(f"Skill preservation check complete!")
        print_info(f"Verdict: {report.verdict.upper()}")
        print_info(f"Overall degradation: {report.overall_degradation:.1f}%")
        
        # Show degraded skills if any
        degraded_skills = [s for s in report.skill_scores if s.status == "degraded"]
        if degraded_skills:
            print_info(f"Degraded skills: {', '.join(s.domain for s in degraded_skills)}")
        
        # Show key recommendations
        if report.recommendations:
            print_info("Key recommendations:")
            for rec in report.recommendations[:3]:  # Show top 3
                print_info(f"  • {rec}")
        
        if output:
            print_info(f"Detailed reports saved to {output}/")
        
        # Exit with appropriate code for automation
        exit_codes = {"safe": 0, "caution": 1, "critical": 2}
        return exit_codes.get(report.verdict, 1)
        
    except ImportError as e:
        print_error(f"Failed to import Neurologist: {e}")
        print_info("This might be due to missing Oumi dependencies")
        return 1
    except Exception as e:
        print_error(f"Skill preservation check failed: {e}")
        logger.error(f"Neurologist error: {e}", exc_info=True)
        return 1


@cli.command()
@click.option("--diagnosis", required=True, help="Path to diagnosis JSON file")
@click.option("--cure-dataset", required=True, help="Path to cure dataset JSONL")
@click.option("--skill-report", help="Path to skill preservation report JSON")
@click.option("--output", default="./healed_model", help="Output directory for trained model")
@click.option("--recipe-name", help="Custom recipe name")
@click.option("--save-dir", default="./recipes", help="Directory to save recipe files")
@click.pass_context
def recipe(ctx, diagnosis, cure_dataset, skill_report, output, recipe_name, save_dir):
    """🔧 Generate training recipe from diagnosis and cure data"""
    logger = ctx.obj['logger']
    
    try:
        from agents.surgeon import Surgeon
        import json
        
        print_info(f"Generating training recipe from diagnosis and cure data...")
        
        # Load diagnosis data
        try:
            with open(diagnosis, 'r') as f:
                diagnosis_data = json.load(f)
        except FileNotFoundError:
            print_error(f"Diagnosis file not found: {diagnosis}")
            return 1
        except json.JSONDecodeError as e:
            print_error(f"Invalid JSON in diagnosis file: {e}")
            return 1
        
        # Load skill preservation data if provided
        skill_data = None
        if skill_report:
            try:
                with open(skill_report, 'r') as f:
                    skill_data = json.load(f)
                print_info(f"Loaded skill preservation report: {skill_report}")
            except FileNotFoundError:
                print_error(f"Skill report file not found: {skill_report}")
                return 1
            except json.JSONDecodeError as e:
                print_error(f"Invalid JSON in skill report: {e}")
                return 1
        
        # Check cure dataset exists
        if not Path(cure_dataset).exists():
            print_error(f"Cure dataset not found: {cure_dataset}")
            return 1
        
        # Initialize surgeon
        surgeon = Surgeon()
        
        # Generate recipe
        recipe_obj = surgeon.generate_recipe(
            diagnosis_data=diagnosis_data,
            cure_dataset_path=cure_dataset,
            skill_preservation_data=skill_data,
            output_dir=output,
            recipe_name=recipe_name
        )
        
        # Save recipe files
        saved_files = surgeon.save_recipe(recipe_obj, save_dir)
        
        print_success(f"Training recipe generated successfully!")
        print_info(f"Recipe ID: {recipe_obj.recipe_id}")
        print_info(f"Severity: {recipe_obj.severity}")
        print_info(f"Model: {recipe_obj.model_name}")
        
        # Show key hyperparameters
        params = recipe_obj.hyperparameters
        print_info(f"Key parameters:")
        print_info(f"  • Learning Rate: {params['learning_rate']:.2e}")
        print_info(f"  • LoRA Rank: {params['lora_r']}")
        print_info(f"  • Epochs: {params['num_epochs']}")
        
        # Show skill adjustments if any
        if recipe_obj.skill_adjustments:
            print_info(f"Skill adjustments applied: {len(recipe_obj.skill_adjustments)}")
        
        print_success(f"Ready to train: oumi train {recipe_obj.recipe_id}.yaml")
        
        return 0
        
    except ImportError as e:
        print_error(f"Failed to import Surgeon: {e}")
        return 1
    except Exception as e:
        print_error(f"Recipe generation failed: {e}")
        logger.error(f"Surgeon error: {e}", exc_info=True)
        return 1


@cli.command()
@click.option("--model", required=True, help="Model ID to treat")
@click.option("--symptom", default="safety", help="Symptom to treat")
@click.option("--output", default="./healed/", help="Output directory")
@click.option("--max-samples", default=50, help="Maximum test samples for diagnosis")
@click.option("--cure-samples", default=100, help="Number of cure examples to generate")
@click.option("--skip-verification", is_flag=True, help="Skip skill preservation verification")
@click.pass_context
def treat(ctx, model, symptom, output, max_samples, cure_samples, skip_verification):
    """🔧 Full treatment pipeline (all 4 agents)"""
    logger = ctx.obj['logger']
    verbose = ctx.obj.get('verbose', False)
    dry_run = ctx.obj.get('dry_run', False)
    json_output = ctx.obj.get('json_output', False)
    
    if dry_run:
        if not json_output:
            hospital_console.console.print("🔧 [bold yellow]DRY RUN[/bold yellow] - Would execute full treatment pipeline:")
            hospital_console.console.print(f"   Model: {model}")
            hospital_console.console.print(f"   Symptom: {symptom}")
            hospital_console.console.print(f"   Output: {output}")
            hospital_console.console.print(f"   Pipeline: Diagnose → Cure → Recipe → {'Verify' if not skip_verification else 'Skip Verify'}")
        else:
            import json
            result = {
                "operation": "treat",
                "dry_run": True,
                "pipeline": ["diagnose", "cure", "recipe", "verify" if not skip_verification else "skip_verify"],
                "parameters": {
                    "model": model,
                    "symptom": symptom,
                    "output": output,
                    "max_samples": max_samples,
                    "cure_samples": cure_samples
                }
            }
            print(json.dumps(result, indent=2))
        return 0
    
    if not json_output:
        hospital_console.console.print(Panel.fit(
            f"🏥 [bold green]FULL TREATMENT PIPELINE[/bold green]\n"
            f"Patient: {model}\n"
            f"Symptom: {symptom}\n"
            f"Treatment Plan: 4-Agent Automated Healing",
            border_style="green"
        ))
    
    treatment_results = {
        "model": model,
        "symptom": symptom,
        "output_dir": output,
        "pipeline_steps": [],
        "status": "in_progress"
    }
    
    try:
        from pathlib import Path
        import tempfile
        import json
        
        # Create output directory
        output_path = Path(output)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Step 1: 🔍 Diagnostician - Diagnose the model
        if not json_output:
            hospital_console.console.print("\n[bold blue]Step 1/4: 🔍 Diagnostician[/bold blue]")
        
        try:
            from agents.diagnostician import Diagnostician
            
            diagnostician = Diagnostician()
            diagnosis = diagnostician.diagnose_model(
                model_id=model,
                symptom=symptom,
                max_samples=max_samples,
                output_dir=str(output_path / "diagnosis")
            )
            
            diagnosis_path = output_path / "diagnosis_results.json"
            with open(diagnosis_path, 'w') as f:
                json.dump(diagnosis.to_dict(), f, indent=2)
            
            treatment_results["pipeline_steps"].append({
                "step": 1,
                "agent": "diagnostician",
                "status": "success",
                "failure_rate": diagnosis.failure_rate,
                "severity": diagnosis.severity,
                "output": str(diagnosis_path)
            })
            
            if not json_output:
                print_success(f"✅ Diagnosis complete: {diagnosis.severity} severity ({diagnosis.failure_rate:.1%} failure rate)")
            
        except Exception as e:
            treatment_results["pipeline_steps"].append({
                "step": 1,
                "agent": "diagnostician",
                "status": "error",
                "error": str(e)
            })
            raise e
        
        # Step 2: 💊 Pharmacist - Generate cure data
        if not json_output:
            hospital_console.console.print("\n[bold green]Step 2/4: 💊 Pharmacist[/bold green]")
        
        try:
            from agents.pharmacist import Pharmacist
            
            pharmacist = Pharmacist()
            cure_result = pharmacist.generate_cure_data(
                diagnosis,
                num_samples=cure_samples,
                output_dir=str(output_path / "cure")
            )
            
            treatment_results["pipeline_steps"].append({
                "step": 2,
                "agent": "pharmacist",
                "status": "success",
                "num_examples": cure_result.num_examples,
                "quality_score": cure_result.quality_score,
                "dataset_path": cure_result.dataset_path
            })
            
            if not json_output:
                print_success(f"✅ Cure data generated: {cure_result.num_examples} examples (quality: {cure_result.quality_score:.2f})")
            
        except Exception as e:
            treatment_results["pipeline_steps"].append({
                "step": 2,
                "agent": "pharmacist",
                "status": "error",
                "error": str(e)
            })
            raise e
        
        # Step 3: 🔧 Surgeon - Generate training recipe
        if not json_output:
            hospital_console.console.print("\n[bold red]Step 3/4: 🔧 Surgeon[/bold red]")
        
        try:
            from agents.surgeon import Surgeon
            
            surgeon = Surgeon()
            recipe = surgeon.generate_recipe(
                diagnosis_data=diagnosis.to_dict(),
                cure_dataset_path=cure_result.dataset_path,
                output_dir=str(output_path / "trained_model")
            )
            
            # Save recipe
            recipe_files = surgeon.save_recipe(recipe, str(output_path / "recipe"))
            
            treatment_results["pipeline_steps"].append({
                "step": 3,
                "agent": "surgeon",
                "status": "success",
                "recipe_id": recipe.recipe_id,
                "recipe_files": recipe_files,
                "hyperparameters": recipe.hyperparameters
            })
            
            if not json_output:
                print_success(f"✅ Training recipe generated: {recipe.recipe_id}")
                hospital_console.console.print(f"   Ready to train: [bold]oumi train {recipe.recipe_id}.yaml[/bold]")
            
        except Exception as e:
            treatment_results["pipeline_steps"].append({
                "step": 3,
                "agent": "surgeon",
                "status": "error",
                "error": str(e)
            })
            raise e
        
        # Step 4: 🧠 Neurologist - Skill preservation (optional)
        if not skip_verification:
            if not json_output:
                hospital_console.console.print("\n[bold purple]Step 4/4: 🧠 Neurologist[/bold purple]")
                hospital_console.console.print("[dim]Note: This would run after training completion[/dim]")
            
            # For now, we'll create a placeholder since we don't have the trained model yet
            treatment_results["pipeline_steps"].append({
                "step": 4,
                "agent": "neurologist",
                "status": "pending",
                "note": "Run after training: oumi-hospital verify --before {model} --after {trained_model_path}"
            })
            
            if not json_output:
                print_info("🧠 Neurologist verification will run after training completion")
        
        treatment_results["status"] = "success"
        
        # Final summary
        if not json_output:
            hospital_console.console.print("\n" + "="*60)
            hospital_console.console.print("[bold green]🎉 TREATMENT PIPELINE COMPLETE[/bold green]")
            hospital_console.console.print("="*60)
            
            hospital_console.console.print(f"\n📋 [bold]Treatment Summary:[/bold]")
            hospital_console.console.print(f"   Patient: {model}")
            hospital_console.console.print(f"   Diagnosis: {diagnosis.severity} {symptom} issues")
            hospital_console.console.print(f"   Cure: {cure_result.num_examples} training examples")
            hospital_console.console.print(f"   Recipe: {recipe.recipe_id}")
            
            hospital_console.console.print(f"\n🚀 [bold]Next Steps:[/bold]")
            hospital_console.console.print(f"   1. Train the model: [bold green]oumi train {recipe.recipe_id}.yaml[/bold green]")
            if not skip_verification:
                hospital_console.console.print(f"   2. Verify skills: [bold blue]oumi-hospital verify --before {model} --after <trained_model>[/bold blue]")
            
            hospital_console.print_community_message()
        
        # Save complete treatment results
        results_path = output_path / "treatment_results.json"
        with open(results_path, 'w') as f:
            json.dump(treatment_results, f, indent=2)
        
        if json_output:
            print(json.dumps(treatment_results, indent=2))
        else:
            print_info(f"📄 Complete results saved to: {results_path}")
        
        return 0
        
    except Exception as e:
        treatment_results["status"] = "error"
        treatment_results["error"] = str(e)
        
        if json_output:
            print(json.dumps(treatment_results, indent=2))
        else:
            print_error(f"Treatment pipeline failed: {e}")
        
        logger.error(f"Treatment pipeline error: {e}", exc_info=True)
        return 1


@cli.command()
@click.option("--recipe", required=True, help="Path to recipe YAML file")
@click.option("--validate-only", is_flag=True, help="Only validate recipe without sharing")
@click.pass_context
def share_recipe(ctx, recipe, validate_only):
    """🌐 Share recipe with Oumi community"""
    logger = ctx.obj['logger']
    verbose = ctx.obj.get('verbose', False)
    dry_run = ctx.obj.get('dry_run', False)
    json_output = ctx.obj.get('json_output', False)
    
    if dry_run:
        if not json_output:
            hospital_console.console.print("🌐 [bold yellow]DRY RUN[/bold yellow] - Would share recipe:")
            hospital_console.console.print(f"   Recipe: {recipe}")
            hospital_console.console.print(f"   Action: {'Validate only' if validate_only else 'Validate and share'}")
        else:
            import json
            result = {
                "operation": "share_recipe",
                "dry_run": True,
                "recipe_path": recipe,
                "validate_only": validate_only
            }
            print(json.dumps(result, indent=2))
        return 0
    
    try:
        import yaml
        from pathlib import Path
        
        recipe_path = Path(recipe)
        
        if not recipe_path.exists():
            if json_output:
                import json
                result = {"operation": "share_recipe", "status": "error", "error": f"Recipe file not found: {recipe}"}
                print(json.dumps(result, indent=2))
            else:
                print_error(f"Recipe file not found: {recipe}")
            return 1
        
        # Validate recipe format
        if not json_output:
            hospital_console.console.print("🔍 Validating recipe format...")
        
        with open(recipe_path, 'r', encoding='utf-8') as f:
            recipe_content = yaml.safe_load(f)
        
        # Check required fields
        required_fields = ["model", "data", "training", "peft"]
        missing_fields = [field for field in required_fields if field not in recipe_content]
        
        if missing_fields:
            if json_output:
                import json
                result = {
                    "operation": "share_recipe",
                    "status": "error",
                    "error": f"Missing required fields: {missing_fields}"
                }
                print(json.dumps(result, indent=2))
            else:
                print_error(f"Recipe validation failed - missing fields: {missing_fields}")
            return 1
        
        # Extract metadata
        metadata = recipe_content.get("_recipe_metadata", {})
        recipe_id = metadata.get("recipe_id", recipe_path.stem)
        
        if not json_output:
            print_success("✅ Recipe validation passed")
            
            # Display recipe info
            hospital_console.console.print(f"\n📋 [bold]Recipe Information:[/bold]")
            hospital_console.console.print(f"   ID: {recipe_id}")
            hospital_console.console.print(f"   Author: {metadata.get('author', 'Unknown')}")
            hospital_console.console.print(f"   Symptom: {metadata.get('symptom', 'Unknown')}")
            hospital_console.console.print(f"   Severity: {metadata.get('severity', 'Unknown')}")
            hospital_console.console.print(f"   Model: {recipe_content['model'].get('model_name', 'Unknown')}")
        
        if validate_only:
            if json_output:
                import json
                result = {
                    "operation": "validate_recipe",
                    "status": "success",
                    "recipe_id": recipe_id,
                    "metadata": metadata
                }
                print(json.dumps(result, indent=2))
            else:
                print_success("✅ Recipe validation complete - ready for sharing")
            return 0
        
        # Create recipes directory structure
        recipes_dir = Path("recipes")
        symptom = metadata.get("symptom", "general")
        symptom_dir = recipes_dir / symptom
        symptom_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy recipe to community structure
        community_recipe_path = symptom_dir / f"{recipe_id}.yaml"
        
        import shutil
        shutil.copy2(recipe_path, community_recipe_path)
        
        # Create recipe index entry
        index_path = recipes_dir / "index.json"
        
        if index_path.exists():
            with open(index_path, 'r') as f:
                index = json.load(f)
        else:
            index = {"recipes": []}
        
        # Add or update recipe entry
        recipe_entry = {
            "recipe_id": recipe_id,
            "path": str(community_recipe_path.relative_to(recipes_dir)),
            "metadata": metadata,
            "shared_at": datetime.now().isoformat()
        }
        
        # Remove existing entry if present
        index["recipes"] = [r for r in index["recipes"] if r["recipe_id"] != recipe_id]
        index["recipes"].append(recipe_entry)
        
        with open(index_path, 'w') as f:
            json.dump(index, f, indent=2)
        
        if json_output:
            import json
            result = {
                "operation": "share_recipe",
                "status": "success",
                "recipe_id": recipe_id,
                "community_path": str(community_recipe_path),
                "metadata": metadata
            }
            print(json.dumps(result, indent=2))
        else:
            print_success(f"🌐 Recipe shared successfully!")
            hospital_console.console.print(f"   Community path: {community_recipe_path}")
            hospital_console.console.print(f"   Recipe ID: {recipe_id}")
            
            hospital_console.print_community_message()
        
        return 0
        
    except yaml.YAMLError as e:
        if json_output:
            import json
            result = {"operation": "share_recipe", "status": "error", "error": f"Invalid YAML: {e}"}
            print(json.dumps(result, indent=2))
        else:
            print_error(f"Invalid YAML format: {e}")
        return 1
    except Exception as e:
        if json_output:
            import json
            result = {"operation": "share_recipe", "status": "error", "error": str(e)}
            print(json.dumps(result, indent=2))
        else:
            print_error(f"Failed to share recipe: {e}")
        logger.error(f"Share recipe error: {e}", exc_info=True)
        return 1


@cli.command()
@click.option("--symptom", help="Filter recipes by symptom (safety, hallucination, bias)")
@click.option("--author", help="Filter recipes by author")
@click.option("--model", help="Filter recipes by base model")
@click.pass_context
def list_recipes(ctx):
    """📋 List available community recipes"""
    logger = ctx.obj['logger']
    verbose = ctx.obj.get('verbose', False)
    json_output = ctx.obj.get('json_output', False)
    
    try:
        from pathlib import Path
        import json
        
        recipes_dir = Path("recipes")
        index_path = recipes_dir / "index.json"
        
        if not index_path.exists():
            if json_output:
                result = {"operation": "list_recipes", "recipes": [], "total": 0}
                print(json.dumps(result, indent=2))
            else:
                print_info("No community recipes found. Share your first recipe with:")
                hospital_console.console.print("   [bold]oumi-hospital share-recipe --recipe <path>[/bold]")
            return 0
        
        with open(index_path, 'r') as f:
            index = json.load(f)
        
        recipes = index.get("recipes", [])
        
        # Apply filters
        if ctx.params.get("symptom"):
            recipes = [r for r in recipes if r.get("metadata", {}).get("symptom") == ctx.params["symptom"]]
        if ctx.params.get("author"):
            recipes = [r for r in recipes if r.get("metadata", {}).get("author") == ctx.params["author"]]
        if ctx.params.get("model"):
            recipes = [r for r in recipes if ctx.params["model"] in r.get("metadata", {}).get("base_model", "")]
        
        if json_output:
            result = {
                "operation": "list_recipes",
                "recipes": recipes,
                "total": len(recipes)
            }
            print(json.dumps(result, indent=2))
        else:
            if not recipes:
                print_info("No recipes match your filters.")
                return 0
            
            hospital_console.console.print(f"\n📋 [bold]Community Recipes ({len(recipes)} found)[/bold]")
            
            # Create recipes table
            from rich.table import Table
            
            table = Table(border_style="blue")
            table.add_column("Recipe ID", style="cyan", no_wrap=True)
            table.add_column("Symptom", style="yellow")
            table.add_column("Severity", style="red")
            table.add_column("Author", style="green")
            table.add_column("Model", style="magenta")
            table.add_column("Shared", style="dim")
            
            for recipe in recipes:
                metadata = recipe.get("metadata", {})
                shared_date = recipe.get("shared_at", "Unknown")
                if shared_date != "Unknown":
                    try:
                        from datetime import datetime
                        shared_date = datetime.fromisoformat(shared_date).strftime("%Y-%m-%d")
                    except:
                        pass
                
                table.add_row(
                    recipe["recipe_id"],
                    metadata.get("symptom", "Unknown"),
                    metadata.get("severity", "Unknown"),
                    metadata.get("author", "Unknown"),
                    metadata.get("base_model", "Unknown"),
                    shared_date
                )
            
            hospital_console.console.print(table)
            
            hospital_console.console.print(f"\n💡 [bold]Usage:[/bold]")
            hospital_console.console.print("   Copy recipe: [bold]cp recipes/<symptom>/<recipe_id>.yaml .[/bold]")
            hospital_console.console.print("   Train model: [bold]oumi train <recipe_id>.yaml[/bold]")
        
        return 0
        
    except Exception as e:
        if json_output:
            import json
            result = {"operation": "list_recipes", "status": "error", "error": str(e)}
            print(json.dumps(result, indent=2))
        else:
            print_error(f"Failed to list recipes: {e}")
        logger.error(f"List recipes error: {e}", exc_info=True)
        return 1


def main():
    """Main entry point"""
    cli()


if __name__ == "__main__":
    main()