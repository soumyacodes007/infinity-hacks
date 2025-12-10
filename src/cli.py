"""
🏥 Oumi Model Hospital - Command Line Interface

Main CLI entry point for the hospital system.
"""

import click
from pathlib import Path
from typing import Optional

from .utils.console import hospital_console, print_header, print_info, print_success
from .utils.logging_config import setup_hospital_logging, get_hospital_logger

# Version info
__version__ = "0.1.0"

@click.group()
@click.version_option(version=__version__)
@click.option("--log-level", default="INFO", help="Logging level")
@click.option("--log-file", help="Log file path")
@click.option("--demo-mode", is_flag=True, help="Enable demo mode (slower output)")
@click.pass_context
def cli(ctx, log_level, log_file, demo_mode):
    """
    🏥 Oumi Model Hospital - Automated AI Model Repair
    
    Don't throw away broken models. Fix them automatically with Oumi.
    """
    # Setup logging
    logger = setup_hospital_logging(log_level, log_file)
    
    # Store context
    ctx.ensure_object(dict)
    ctx.obj['logger'] = logger
    ctx.obj['demo_mode'] = demo_mode
    
    # Print header
    print_header(
        "Oumi Model Hospital", 
        "Automated AI Model Diagnosis, Repair, and Validation"
    )
    
    if demo_mode:
        hospital_console.print_demo_banner()


@cli.command()
@click.option("--model", required=True, help="Model ID to diagnose")
@click.option("--symptom", default="safety", help="Symptom to test (safety, hallucination, bias)")
@click.option("--output", help="Output directory for results")
@click.pass_context
def diagnose(ctx, model, symptom, output):
    """🔍 Diagnose model issues using red-team attacks"""
    logger = ctx.obj['logger']
    
    print_info(f"Starting diagnosis of {model} for {symptom} issues...")
    
    # TODO: Implement diagnostician agent
    print_info("Diagnostician agent not yet implemented")
    print_info("This will be implemented in Task 3")


@cli.command()
@click.option("--diagnosis", required=True, help="Path to diagnosis results")
@click.option("--output", help="Output path for cure dataset")
@click.pass_context  
def cure(ctx, diagnosis, output):
    """💊 Generate cure data from diagnosis results"""
    logger = ctx.obj['logger']
    
    print_info(f"Generating cure data from {diagnosis}...")
    
    # TODO: Implement pharmacist agent
    print_info("Pharmacist agent not yet implemented")
    print_info("This will be implemented in Task 4")


@cli.command()
@click.option("--model-before", required=True, help="Original model path/ID")
@click.option("--model-after", required=True, help="Healed model path")
@click.pass_context
def verify(ctx, model_before, model_after):
    """🧠 Verify skill preservation after treatment"""
    logger = ctx.obj['logger']
    
    print_info(f"Checking skill preservation: {model_before} → {model_after}")
    
    # TODO: Implement neurologist agent
    print_info("Neurologist agent not yet implemented") 
    print_info("This will be implemented in Task 5")


@cli.command()
@click.option("--model", required=True, help="Model ID to treat")
@click.option("--symptom", default="safety", help="Symptom to treat")
@click.option("--output", default="./healed/", help="Output directory")
@click.pass_context
def treat(ctx, model, symptom, output):
    """🔧 Full treatment pipeline (all agents)"""
    logger = ctx.obj['logger']
    
    print_info(f"Starting full treatment of {model}...")
    
    # This will orchestrate all 4 agents
    print_info("Full treatment pipeline not yet implemented")
    print_info("This will chain all agents together")
    
    hospital_console.print_community_message()


@cli.command()
@click.option("--recipe", required=True, help="Path to recipe YAML")
@click.pass_context
def share_recipe(ctx, recipe):
    """🌐 Share recipe with Oumi community"""
    logger = ctx.obj['logger']
    
    print_info(f"Sharing recipe {recipe} with community...")
    
    # TODO: Implement community sharing
    print_info("Community sharing not yet implemented")
    print_info("This will validate and upload recipes")


@cli.command()
@click.pass_context
def list_recipes(ctx):
    """📋 List available community recipes"""
    logger = ctx.obj['logger']
    
    print_info("Listing community recipes...")
    
    # TODO: Implement recipe listing
    print_info("Recipe listing not yet implemented")


def main():
    """Main entry point"""
    cli()


if __name__ == "__main__":
    main()