#!/usr/bin/env python3
"""
HuggingFace Model Fine-Tuning CLI

Command-line interface for automated model training.

Usage:
    python cli.py train --config config.yaml
    python cli.py batch --configs config1.yaml config2.yaml
    python cli.py resume --checkpoint path/to/checkpoint --config config.yaml
    python cli.py generate-config --model path/to/model
    python cli.py validate --config config.yaml
    python cli.py info
"""

import argparse
import sys
from pathlib import Path


def cmd_train(args):
    """Train a model using a configuration file"""
    from trainer import ConfigLoader, ModelTrainer
    
    print(f"Loading configuration from: {args.config}")
    
    try:
        config = ConfigLoader.load(args.config)
    except Exception as e:
        print(f"ERROR: Failed to load configuration: {e}")
        sys.exit(1)
    
    print(f"Starting training: {config.output.experiment_name}")
    print(f"  Model: {config.model.base_model}")
    print(f"  Dataset: {config.dataset.name}")
    print(f"  Output: {config.output.model_dir}")
    
    if config.output.hub_repo:
        print(f"  Hub: {config.output.hub_repo}")
    
    print()
    
    try:
        trainer = ModelTrainer(config)
        trainer.run()
        print("\nTraining completed successfully!")
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nERROR: Training failed: {e}")
        sys.exit(1)


def cmd_batch(args):
    """Run batch training with multiple configurations"""
    from trainer import BatchTrainer
    
    print(f"Starting batch training with {len(args.configs)} configurations...")
    
    # Validate all configs exist
    for config_path in args.configs:
        if not Path(config_path).exists():
            print(f"ERROR: Configuration file not found: {config_path}")
            sys.exit(1)
    
    try:
        batch_trainer = BatchTrainer(args.configs)
        batch_trainer.run()
    except KeyboardInterrupt:
        print("\nBatch training interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nERROR: Batch training failed: {e}")
        sys.exit(1)


def cmd_resume(args):
    """Resume training from a checkpoint"""
    from trainer import ConfigLoader, ModelTrainer
    
    if not Path(args.checkpoint).exists():
        print(f"ERROR: Checkpoint not found: {args.checkpoint}")
        sys.exit(1)
    
    print(f"Loading configuration from: {args.config}")
    print(f"Resuming from checkpoint: {args.checkpoint}")
    
    try:
        config = ConfigLoader.load(args.config)
        trainer = ModelTrainer(config)
        trainer.run(resume_from_checkpoint=args.checkpoint)
        print("\nTraining resumed and completed successfully!")
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nERROR: Resume failed: {e}")
        sys.exit(1)


def cmd_generate_config(args):
    """Generate config.json from a trained model"""
    from trainer import generate_config_from_model
    
    output = args.output or "config.json"
    
    print(f"Generating config from: {args.model}")
    print(f"Output: {output}")
    
    try:
        generate_config_from_model(args.model, output)
        print(f"\nConfig generated successfully: {output}")
    except Exception as e:
        print(f"\nERROR: Config generation failed: {e}")
        sys.exit(1)


def cmd_validate(args):
    """Validate a configuration file"""
    from trainer import ConfigLoader
    
    print(f"Validating configuration: {args.config}")
    print()
    
    if not Path(args.config).exists():
        print(f"ERROR: Configuration file not found: {args.config}")
        sys.exit(1)
    
    issues = ConfigLoader.validate_config_file(args.config)
    
    if not issues:
        print("Configuration is valid!")
        
        # Try to load and show summary
        try:
            config = ConfigLoader.load(args.config)
            print("\nConfiguration Summary:")
            print(f"  Model: {config.model.base_model}")
            print(f"  Dataset: {config.dataset.name}")
            print(f"  LoRA r: {config.lora.r}, alpha: {config.lora.lora_alpha}")
            print(f"  Target modules: {config.lora.target_modules}")
            print(f"  Epochs: {config.training.num_epochs}")
            print(f"  Learning rate: {config.training.learning_rate}")
            print(f"  Batch size: {config.training.per_device_batch_size}")
            print(f"  Output: {config.output.model_dir}")
            if config.output.hub_repo:
                print(f"  Hub repo: {config.output.hub_repo}")
        except Exception as e:
            print(f"\nWarning: Could not load full config: {e}")
        
        sys.exit(0)
    else:
        print("Issues found:")
        for issue in issues:
            print(f"  {issue}")
        
        errors = [i for i in issues if i.startswith("ERROR")]
        if errors:
            sys.exit(1)
        else:
            print("\nConfiguration has warnings but is valid.")
            sys.exit(0)


def cmd_info(args):
    """Display system information"""
    import torch
    
    print("HuggingFace Model Fine-Tuning Toolkit")
    print("=" * 40)
    print()
    
    print("System Information:")
    print(f"  Python: {sys.version.split()[0]}")
    print(f"  PyTorch: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"  CUDA version: {torch.version.cuda}")
        print(f"  GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            name = torch.cuda.get_device_name(i)
            memory = torch.cuda.get_device_properties(i).total_memory / 1e9
            print(f"    [{i}] {name} ({memory:.1f} GB)")
    
    print()
    print("Installed packages:")
    
    packages = [
        ("transformers", "transformers"),
        ("peft", "peft"),
        ("datasets", "datasets"),
        ("bitsandbytes", "bitsandbytes"),
        ("accelerate", "accelerate"),
    ]
    
    for display_name, import_name in packages:
        try:
            module = __import__(import_name)
            version = getattr(module, "__version__", "unknown")
            print(f"  {display_name}: {version}")
        except ImportError:
            print(f"  {display_name}: NOT INSTALLED")
    
    print()
    print("Usage:")
    print("  python cli.py train --config config.yaml")
    print("  python cli.py batch --configs config1.yaml config2.yaml")
    print("  python cli.py resume --checkpoint path/checkpoint --config config.yaml")
    print("  python cli.py generate-config --model path/to/model")
    print("  python cli.py validate --config config.yaml")


def main():
    parser = argparse.ArgumentParser(
        description="HuggingFace Model Fine-Tuning CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train with a configuration file
  python cli.py train --config configs/example_minimal.yaml

  # Run batch training
  python cli.py batch --configs config1.yaml config2.yaml config3.yaml

  # Resume from checkpoint
  python cli.py resume --checkpoint experiments/*/checkpoints/checkpoint-100 --config config.yaml

  # Generate config.json from trained model
  python cli.py generate-config --model akumar23/mental-falcon-7b

  # Validate configuration
  python cli.py validate --config my_config.yaml

  # Show system info
  python cli.py info
"""
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Train a model using a configuration file")
    train_parser.add_argument(
        "--config", "-c",
        required=True,
        help="Path to YAML configuration file"
    )
    train_parser.set_defaults(func=cmd_train)
    
    # Batch command
    batch_parser = subparsers.add_parser("batch", help="Run batch training with multiple configurations")
    batch_parser.add_argument(
        "--configs", "-c",
        nargs="+",
        required=True,
        help="Paths to YAML configuration files"
    )
    batch_parser.set_defaults(func=cmd_batch)
    
    # Resume command
    resume_parser = subparsers.add_parser("resume", help="Resume training from a checkpoint")
    resume_parser.add_argument(
        "--checkpoint", "-p",
        required=True,
        help="Path to checkpoint directory"
    )
    resume_parser.add_argument(
        "--config", "-c",
        required=True,
        help="Path to YAML configuration file"
    )
    resume_parser.set_defaults(func=cmd_resume)
    
    # Generate config command
    genconfig_parser = subparsers.add_parser(
        "generate-config",
        help="Generate config.json from a trained model"
    )
    genconfig_parser.add_argument(
        "--model", "-m",
        required=True,
        help="Path or HuggingFace repo of trained model"
    )
    genconfig_parser.add_argument(
        "--output", "-o",
        default="config.json",
        help="Output path for config.json (default: config.json)"
    )
    genconfig_parser.set_defaults(func=cmd_generate_config)
    
    # Validate command
    validate_parser = subparsers.add_parser("validate", help="Validate a configuration file")
    validate_parser.add_argument(
        "--config", "-c",
        required=True,
        help="Path to YAML configuration file"
    )
    validate_parser.set_defaults(func=cmd_validate)
    
    # Info command
    info_parser = subparsers.add_parser("info", help="Display system information")
    info_parser.set_defaults(func=cmd_info)
    
    # Parse arguments
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        sys.exit(0)
    
    # Execute command
    args.func(args)


if __name__ == "__main__":
    main()
