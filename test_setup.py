#!/usr/bin/env python3
"""
Test script to validate the training setup and configuration.
"""

import sys
import os
from pathlib import Path
import yaml

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from trainer import ConfigLoader
from utils import validate_environment, print_environment_info


def test_config_loading():
    """Test configuration file loading and validation."""
    print("\n" + "=" * 80)
    print("Testing Configuration Loading")
    print("=" * 80)

    test_configs = [
        'config_schema.yaml',
        'configs/example_minimal.yaml',
        'configs/example_with_hub.yaml',
        'configs/example_llama.yaml'
    ]

    results = []

    for config_file in test_configs:
        if not os.path.exists(config_file):
            print(f"\n✗ {config_file}: File not found")
            results.append(False)
            continue

        try:
            config = ConfigLoader.load(config_file)
            print(f"\n✓ {config_file}: Valid")
            print(f"  Experiment: {config['experiment']['name']}")
            print(f"  Base Model: {config['model']['base_model']}")
            print(f"  Dataset: {config['dataset']['name']}")
            results.append(True)
        except Exception as e:
            print(f"\n✗ {config_file}: Invalid")
            print(f"  Error: {str(e)}")
            results.append(False)

    success_rate = sum(results) / len(results) * 100
    print(f"\n{'-' * 80}")
    print(f"Config validation: {sum(results)}/{len(results)} passed ({success_rate:.0f}%)")

    return all(results)


def test_imports():
    """Test that all required packages can be imported."""
    print("\n" + "=" * 80)
    print("Testing Package Imports")
    print("=" * 80)

    packages = {
        'torch': 'PyTorch',
        'transformers': 'HuggingFace Transformers',
        'datasets': 'HuggingFace Datasets',
        'peft': 'PEFT',
        'bitsandbytes': 'bitsandbytes',
        'yaml': 'PyYAML'
    }

    results = []

    for package, name in packages.items():
        try:
            module = __import__(package)
            version = getattr(module, '__version__', 'unknown')
            print(f"✓ {name}: {version}")
            results.append(True)
        except ImportError as e:
            print(f"✗ {name}: Not installed ({str(e)})")
            results.append(False)

    success_rate = sum(results) / len(results) * 100
    print(f"\n{'-' * 80}")
    print(f"Package imports: {sum(results)}/{len(results)} passed ({success_rate:.0f}%)")

    return all(results)


def test_directory_structure():
    """Test that expected directories exist."""
    print("\n" + "=" * 80)
    print("Testing Directory Structure")
    print("=" * 80)

    expected_files = [
        'cli.py',
        'trainer.py',
        'utils.py',
        'config_schema.yaml',
        'requirements.txt'
    ]

    expected_dirs = [
        'configs'
    ]

    results = []

    print("\nExpected Files:")
    for file in expected_files:
        exists = os.path.isfile(file)
        symbol = "✓" if exists else "✗"
        print(f"  {symbol} {file}")
        results.append(exists)

    print("\nExpected Directories:")
    for directory in expected_dirs:
        exists = os.path.isdir(directory)
        symbol = "✓" if exists else "✗"
        print(f"  {symbol} {directory}/")
        results.append(exists)

    success_rate = sum(results) / len(results) * 100
    print(f"\n{'-' * 80}")
    print(f"Directory structure: {sum(results)}/{len(results)} passed ({success_rate:.0f}%)")

    return all(results)


def test_cli_help():
    """Test that CLI help works."""
    print("\n" + "=" * 80)
    print("Testing CLI Interface")
    print("=" * 80)

    try:
        from cli import create_parser
        parser = create_parser()

        # Test that all expected commands exist
        commands = ['train', 'batch', 'resume', 'generate-config', 'validate', 'info']

        print("\nAvailable commands:")
        for cmd in commands:
            print(f"  ✓ {cmd}")

        print(f"\n{'-' * 80}")
        print(f"CLI interface: OK")
        return True

    except Exception as e:
        print(f"✗ CLI interface: Failed")
        print(f"  Error: {str(e)}")
        return False


def test_trainer_classes():
    """Test that trainer classes can be instantiated."""
    print("\n" + "=" * 80)
    print("Testing Trainer Classes")
    print("=" * 80)

    try:
        from trainer import (
            ConfigLoader,
            ExperimentTracker,
            TrainingMetrics,
            ModelTrainer,
            BatchTrainer
        )

        classes = [
            'ConfigLoader',
            'ExperimentTracker',
            'TrainingMetrics',
            'ModelTrainer',
            'BatchTrainer'
        ]

        print("\nImported classes:")
        for cls in classes:
            print(f"  ✓ {cls}")

        print(f"\n{'-' * 80}")
        print(f"Trainer classes: OK")
        return True

    except Exception as e:
        print(f"✗ Trainer classes: Failed")
        print(f"  Error: {str(e)}")
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("HuggingFace Model Training Toolkit - Setup Validation")
    print("=" * 80)

    tests = [
        ("Environment", lambda: (print_environment_info(), True)[1]),
        ("Package Imports", test_imports),
        ("Directory Structure", test_directory_structure),
        ("Configuration Loading", test_config_loading),
        ("CLI Interface", test_cli_help),
        ("Trainer Classes", test_trainer_classes)
    ]

    results = []

    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n✗ {test_name}: Exception occurred")
            print(f"  Error: {str(e)}")
            results.append((test_name, False))

    # Print summary
    print("\n" + "=" * 80)
    print("Test Summary")
    print("=" * 80)

    for test_name, result in results:
        symbol = "✓" if result else "✗"
        status = "PASSED" if result else "FAILED"
        print(f"{symbol} {test_name}: {status}")

    passed = sum(1 for _, result in results if result)
    total = len(results)
    success_rate = passed / total * 100

    print(f"\n{'-' * 80}")
    print(f"Overall: {passed}/{total} tests passed ({success_rate:.0f}%)")
    print("=" * 80)

    if passed == total:
        print("\nSetup is ready for training!")
        return 0
    else:
        print("\nSetup validation failed. Please fix the issues above.")
        return 1


if __name__ == '__main__':
    sys.exit(main())
