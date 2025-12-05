"""
Utility functions for model training and evaluation.
"""

import os
import json
import torch
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel, PeftConfig


def load_model_for_inference(
    model_path: str,
    device_map: str = "auto",
    load_in_4bit: bool = True
) -> Tuple[Any, Any]:
    """
    Load a trained PEFT model for inference.

    Args:
        model_path: Path to the trained PEFT model
        device_map: Device mapping strategy
        load_in_4bit: Whether to use 4-bit quantization

    Returns:
        Tuple of (model, tokenizer)
    """
    # Setup quantization config
    bnb_config = None
    if load_in_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )

    # Load PEFT config to get base model
    peft_config = PeftConfig.from_pretrained(model_path)
    base_model_name = peft_config.base_model_name_or_path

    print(f"Loading base model: {base_model_name}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=bnb_config,
        device_map=device_map,
        trust_remote_code=True
    )

    print(f"Loading PEFT adapter from: {model_path}")
    model = PeftModel.from_pretrained(base_model, model_path)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Model and tokenizer loaded successfully")
    return model, tokenizer


def generate_text(
    model: Any,
    tokenizer: Any,
    prompt: str,
    max_new_tokens: int = 200,
    temperature: float = 0.7,
    top_p: float = 0.9,
    do_sample: bool = True,
    num_return_sequences: int = 1
) -> List[str]:
    """
    Generate text using a trained model.

    Args:
        model: Loaded model
        tokenizer: Loaded tokenizer
        prompt: Input prompt text
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_p: Nucleus sampling parameter
        do_sample: Whether to use sampling
        num_return_sequences: Number of sequences to generate

    Returns:
        List of generated texts
    """
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            num_return_sequences=num_return_sequences,
            pad_token_id=tokenizer.eos_token_id
        )

    # Decode outputs
    generated_texts = []
    for output in outputs:
        text = tokenizer.decode(output, skip_special_tokens=True)
        # Remove the input prompt from output
        if text.startswith(prompt):
            text = text[len(prompt):].strip()
        generated_texts.append(text)

    return generated_texts


def count_model_parameters(model: Any) -> Dict[str, int]:
    """
    Count trainable and total parameters in a model.

    Args:
        model: PyTorch model

    Returns:
        Dictionary with parameter counts
    """
    trainable_params = 0
    all_params = 0

    for _, param in model.named_parameters():
        num_params = param.numel()
        all_params += num_params
        if param.requires_grad:
            trainable_params += num_params

    return {
        'trainable_params': trainable_params,
        'total_params': all_params,
        'trainable_percentage': 100 * trainable_params / all_params if all_params > 0 else 0
    }


def estimate_gpu_memory(
    model_size_gb: float,
    batch_size: int = 1,
    sequence_length: int = 2048,
    use_4bit: bool = True,
    gradient_checkpointing: bool = True
) -> Dict[str, float]:
    """
    Estimate GPU memory requirements for training.

    Args:
        model_size_gb: Base model size in GB
        batch_size: Training batch size
        sequence_length: Maximum sequence length
        use_4bit: Whether using 4-bit quantization
        gradient_checkpointing: Whether using gradient checkpointing

    Returns:
        Dictionary with memory estimates
    """
    # Base model memory
    model_memory = model_size_gb
    if use_4bit:
        model_memory *= 0.25  # 4-bit reduces to ~25%

    # Optimizer states (usually 2x model size for Adam)
    optimizer_memory = model_memory * 2

    # Gradients
    gradient_memory = model_memory
    if gradient_checkpointing:
        gradient_memory *= 0.5  # Checkpoint reduces gradient memory

    # Activations (depends on batch size and sequence length)
    activation_memory = (batch_size * sequence_length * 4) / 1e9  # Rough estimate

    total_memory = model_memory + optimizer_memory + gradient_memory + activation_memory

    return {
        'model_memory_gb': model_memory,
        'optimizer_memory_gb': optimizer_memory,
        'gradient_memory_gb': gradient_memory,
        'activation_memory_gb': activation_memory,
        'total_estimated_gb': total_memory,
        'recommended_gpu_gb': total_memory * 1.2  # Add 20% buffer
    }


def find_latest_checkpoint(experiment_dir: str) -> Optional[str]:
    """
    Find the latest checkpoint in an experiment directory.

    Args:
        experiment_dir: Path to experiment directory

    Returns:
        Path to latest checkpoint or None
    """
    checkpoints_dir = Path(experiment_dir) / "checkpoints"

    if not checkpoints_dir.exists():
        return None

    # Find all checkpoint directories
    checkpoints = [
        d for d in checkpoints_dir.iterdir()
        if d.is_dir() and d.name.startswith('checkpoint-')
    ]

    if not checkpoints:
        return None

    # Sort by checkpoint number
    checkpoints.sort(key=lambda x: int(x.name.split('-')[1]))

    return str(checkpoints[-1])


def load_training_metrics(experiment_dir: str) -> Optional[Dict[str, Any]]:
    """
    Load training metrics from an experiment directory.

    Args:
        experiment_dir: Path to experiment directory

    Returns:
        Dictionary with metrics or None
    """
    metrics_file = Path(experiment_dir) / "metrics" / "training_metrics.json"

    if not metrics_file.exists():
        return None

    with open(metrics_file, 'r') as f:
        return json.load(f)


def compare_experiments(experiment_dirs: List[str]) -> Dict[str, Any]:
    """
    Compare metrics across multiple experiments.

    Args:
        experiment_dirs: List of experiment directory paths

    Returns:
        Dictionary with comparison data
    """
    comparison = {
        'experiments': [],
        'best_loss': None,
        'fastest_training': None
    }

    for exp_dir in experiment_dirs:
        metrics = load_training_metrics(exp_dir)
        if metrics:
            comparison['experiments'].append({
                'name': metrics.get('experiment_name'),
                'final_loss': metrics.get('final_loss'),
                'training_time': metrics.get('total_training_time'),
                'trainable_params': metrics.get('trainable_params'),
                'path': exp_dir
            })

    # Find best experiment by loss
    if comparison['experiments']:
        comparison['best_loss'] = min(
            comparison['experiments'],
            key=lambda x: x['final_loss'] if x['final_loss'] else float('inf')
        )

        comparison['fastest_training'] = min(
            comparison['experiments'],
            key=lambda x: x['training_time'] if x['training_time'] else float('inf')
        )

    return comparison


def validate_environment() -> Dict[str, bool]:
    """
    Validate that the environment is properly set up for training.

    Returns:
        Dictionary with validation results
    """
    results = {
        'cuda_available': torch.cuda.is_available(),
        'gpu_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        'transformers_installed': False,
        'peft_installed': False,
        'datasets_installed': False,
        'bitsandbytes_installed': False
    }

    # Check required packages
    try:
        import transformers
        results['transformers_installed'] = True
        results['transformers_version'] = transformers.__version__
    except ImportError:
        pass

    try:
        import peft
        results['peft_installed'] = True
        results['peft_version'] = peft.__version__
    except ImportError:
        pass

    try:
        import datasets
        results['datasets_installed'] = True
        results['datasets_version'] = datasets.__version__
    except ImportError:
        pass

    try:
        import bitsandbytes
        results['bitsandbytes_installed'] = True
        results['bitsandbytes_version'] = bitsandbytes.__version__
    except ImportError:
        pass

    # Add GPU info if available
    if results['cuda_available']:
        results['gpu_name'] = torch.cuda.get_device_name(0)
        results['gpu_memory_gb'] = torch.cuda.get_device_properties(0).total_memory / 1e9

    return results


def print_environment_info():
    """Print formatted environment information."""
    results = validate_environment()

    print("=" * 80)
    print("Environment Validation")
    print("=" * 80)

    # CUDA info
    cuda_status = "Available" if results['cuda_available'] else "Not Available"
    print(f"\nCUDA: {cuda_status}")

    if results['cuda_available']:
        print(f"  GPU Count: {results['gpu_count']}")
        print(f"  GPU: {results.get('gpu_name', 'Unknown')}")
        print(f"  Memory: {results.get('gpu_memory_gb', 0):.2f} GB")

    # Package info
    print("\nRequired Packages:")
    packages = ['transformers', 'peft', 'datasets', 'bitsandbytes']

    for package in packages:
        installed = results.get(f'{package}_installed', False)
        status = "Installed" if installed else "Missing"
        version = results.get(f'{package}_version', '')

        symbol = "✓" if installed else "✗"
        print(f"  {symbol} {package}: {status} {version}")

    print("\n" + "=" * 80)

    # Check if environment is ready
    all_packages = all(results.get(f'{p}_installed', False) for p in packages)
    if results['cuda_available'] and all_packages:
        print("Environment is ready for training!")
    else:
        print("Environment setup incomplete:")
        if not results['cuda_available']:
            print("  - CUDA is not available")
        if not all_packages:
            print("  - Some required packages are missing")

    print("=" * 80)


if __name__ == '__main__':
    # When run directly, print environment info
    print_environment_info()
