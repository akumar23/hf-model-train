# HuggingFace Model Fine-Tuning Toolkit

A comprehensive toolkit for fine-tuning large language models using Parameter-Efficient Fine-Tuning (PEFT) with LoRA and 4-bit quantization. This toolkit makes it easy to customize pre-trained models for your specific use cases without requiring massive computational resources.

## Table of Contents

- [What is Fine-Tuning?](#what-is-fine-tuning)
- [Key Features](#key-features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [CLI Commands](#cli-commands)
- [Configuration System](#configuration-system)
- [Experiment Tracking](#experiment-tracking)
- [Supported Architectures](#supported-architectures)
- [Troubleshooting](#troubleshooting)
- [Legacy Scripts](#legacy-scripts)
- [Technical Details](#technical-details)

## What is Fine-Tuning?

**Fine-tuning** is the process of taking a pre-trained language model (like Falcon, LLaMA, or Mistral) and training it further on your specific dataset. This allows you to:

- Adapt the model to your domain (e.g., medical, legal, customer support)
- Teach it your organization's specific knowledge
- Customize its response style and behavior
- Improve performance on specialized tasks

### Why LoRA and Quantization?

Training large language models requires enormous computational resources. This toolkit uses two key techniques to make fine-tuning accessible:

- **LoRA (Low-Rank Adaptation)**: Instead of updating all model parameters, LoRA only trains small "adapter" layers, reducing memory requirements by up to 90% while maintaining quality.

- **4-bit Quantization**: Compresses the model weights from 32-bit to 4-bit precision, allowing you to train 7B parameter models on consumer GPUs with 12-16GB VRAM.

Together, these techniques enable fine-tuning billion-parameter models on a single GPU.

## Key Features

- **YAML Configuration System** - Define all training parameters in easy-to-read configuration files
- **Experiment Tracking** - Automatic logging, metrics tracking, and organized output directories
- **Checkpoint Resume** - Continue interrupted training from the last saved checkpoint
- **Batch Training** - Run multiple experiments sequentially with different configurations
- **Configuration Validation** - Verify your config files before starting long training runs
- **HuggingFace Hub Integration** - Automatically push trained models to HuggingFace Hub
- **Multiple Architecture Support** - Works with Falcon, LLaMA, Mistral, GPT-2, and more
- **Memory Efficient** - Train 7B models on GPUs with 12GB+ VRAM

## Requirements

- **Python**: 3.11.5 (other versions may work but are untested)
- **Hardware**: CUDA-enabled NVIDIA GPU with 12GB+ VRAM recommended
- **Operating System**: Linux or macOS (Windows with WSL2)

### GPU Memory Requirements

| Model Size | Minimum VRAM | Recommended VRAM |
|------------|--------------|------------------|
| 1-3B params | 8GB | 12GB |
| 7B params | 12GB | 16GB |
| 13B params | 24GB | 32GB |

## Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd hf-model-train
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify installation**
   ```bash
   python cli.py info
   ```

   This should display your system information, Python version, PyTorch version, and GPU availability.

## Quick Start

The fastest way to start fine-tuning is using the minimal example configuration:

1. **Create a configuration file** (or use `configs/example_minimal.yaml`):
   ```yaml
   model:
     base_model: "vilsonrodrigues/falcon-7b-instruct-sharded"

   dataset:
     name: "Amod/mental_health_counseling_conversations"
     prompt_template:
       format: "<human>: {User}\n<assistant>: {Prompt}"
       context_column: "Context"

   output:
     model_dir: "my-finetuned-model"
     experiment_name: "minimal-training"
   ```

2. **Start training**
   ```bash
   python cli.py train --config configs/example_minimal.yaml
   ```

3. **Monitor progress**

   Training logs are displayed in real-time and saved to:
   ```
   experiments/YYYYMMDD_HHMMSS_minimal-training/logs/training.log
   ```

4. **Find your trained model**

   After training completes, your model will be saved to:
   ```
   experiments/YYYYMMDD_HHMMSS_minimal-training/my-finetuned-model/
   ```

That's it! You've fine-tuned your first model.

## CLI Commands

The toolkit provides a command-line interface with several commands:

### Train

Train a model using a configuration file.

```bash
python cli.py train --config <path-to-config.yaml>
```

**Example:**
```bash
python cli.py train --config configs/example_minimal.yaml
```

### Batch

Run multiple training configurations sequentially. Useful for hyperparameter sweeps or training multiple models overnight.

```bash
python cli.py batch --configs <config1.yaml> <config2.yaml> <config3.yaml>
```

**Example:**
```bash
python cli.py batch --configs configs/experiment1.yaml configs/experiment2.yaml configs/experiment3.yaml
```

### Resume

Continue training from a saved checkpoint. Useful if training was interrupted.

```bash
python cli.py resume --checkpoint <checkpoint-path> --config <config.yaml>
```

**Example:**
```bash
python cli.py resume --checkpoint experiments/20241205_143022_my-experiment/checkpoints/checkpoint-500 --config configs/my_config.yaml
```

### Validate

Check if a configuration file is valid before starting training.

```bash
python cli.py validate --config <config.yaml>
```

**Example:**
```bash
python cli.py validate --config configs/my_config.yaml
```

### Generate Config

Generate a `config.json` file for a trained PEFT model. This is required for compatibility with HuggingFace Transformers and Tokenizers.

```bash
python cli.py generate-config --model <huggingface-repo-or-local-path>
```

**Examples:**
```bash
# From HuggingFace Hub
python cli.py generate-config --model akumar23/mental-falcon-7b

# From local directory
python cli.py generate-config --model ./experiments/20241205_143022_my-experiment/my-finetuned-model
```

### Info

Display system information including Python version, PyTorch version, CUDA availability, and GPU details.

```bash
python cli.py info
```

## Configuration System

The toolkit uses YAML files for configuration, making it easy to define, version control, and share training setups.

### Configuration File Structure

A complete configuration file has the following sections:

```yaml
# Model Configuration
model:
  base_model: "vilsonrodrigues/falcon-7b-instruct-sharded"
  trust_remote_code: true
  device_map: "auto"

# Quantization Settings (for memory efficiency)
quantization:
  load_in_4bit: true
  use_double_quant: true
  quant_type: "nf4"
  compute_dtype: "float16"

# LoRA Configuration (adapter parameters)
lora:
  r: 16
  lora_alpha: 32
  target_modules: ["query_key_value"]
  lora_dropout: 0.05
  bias: "none"
  task_type: "CAUSAL_LM"

# Dataset Configuration
dataset:
  name: "Amod/mental_health_counseling_conversations"
  train_split: "train"
  prompt_template:
    format: "<human>: {User}\n<assistant>: {Prompt}"
    context_column: "Context"
  max_length: 512
  truncation: true
  padding: "max_length"

# Training Parameters
training:
  num_epochs: 3
  per_device_batch_size: 4
  gradient_accumulation_steps: 4
  learning_rate: 2.0e-4
  lr_scheduler_type: "cosine"
  warmup_steps: 100
  weight_decay: 0.01
  max_grad_norm: 1.0
  optimizer: "paged_adamw_8bit"
  save_steps: 100
  logging_steps: 10
  eval_steps: 100

# Generation Settings (for testing during training)
generation:
  max_new_tokens: 128
  temperature: 0.7
  top_p: 0.95
  top_k: 50

# Output Configuration
output:
  model_dir: "my-finetuned-model"
  experiments_dir: "experiments"
  experiment_name: "my-experiment"
  hub_repo: null  # Set to "username/repo-name" to push to Hub
  generate_config: true

# Environment Settings
environment:
  cuda_visible_devices: null  # e.g., "0,1" to use specific GPUs
  seed: 42

# Logging Configuration
logging:
  level: "INFO"
  log_to_file: true
  log_to_console: true
```

### Example Configurations

The toolkit includes several example configurations:

1. **configs/example_minimal.yaml** - Quick start template with minimal required settings
2. **configs/example_with_hub.yaml** - Production template with HuggingFace Hub push enabled
3. **configs/example_llama.yaml** - Configuration optimized for LLaMA/Mistral models

### Configuration Reference

For a complete reference of all available configuration options, see `config_schema.yaml` in the repository root.

#### Key Configuration Options

**Model Section:**
- `base_model`: HuggingFace model ID or local path (required)
- `trust_remote_code`: Allow custom model code execution (default: true)
- `device_map`: Device allocation strategy (default: "auto")

**LoRA Section:**
- `r`: LoRA rank - higher values = more parameters but better quality (default: 16)
- `lora_alpha`: LoRA scaling factor (default: 32)
- `target_modules`: Which model layers to apply LoRA to (architecture-specific)
- `lora_dropout`: Dropout probability for LoRA layers (default: 0.05)

**Dataset Section:**
- `name`: HuggingFace dataset ID or local path (required)
- `prompt_template.format`: Template string for formatting examples (required)
- `max_length`: Maximum sequence length in tokens (default: 512)

**Training Section:**
- `num_epochs`: Number of training epochs (default: 3)
- `per_device_batch_size`: Batch size per GPU (default: 4)
- `learning_rate`: Peak learning rate (default: 2e-4)
- `gradient_accumulation_steps`: Steps to accumulate before updating (default: 4)

**Output Section:**
- `model_dir`: Output directory name for the trained model (required)
- `experiment_name`: Name for the experiment (used in directory naming)
- `hub_repo`: HuggingFace Hub repository to push to (optional)

## Experiment Tracking

Every training run creates a timestamped experiment directory with complete logs and artifacts:

```
experiments/YYYYMMDD_HHMMSS_experiment-name/
├── config.yaml                      # Complete configuration used
├── logs/
│   └── training.log                 # Detailed training logs
├── metrics/
│   └── training_metrics.json        # Loss, learning rate, etc.
├── checkpoints/
│   ├── checkpoint-100/              # Periodic checkpoints
│   ├── checkpoint-200/
│   └── checkpoint-300/
└── my-finetuned-model/              # Final trained model
    ├── adapter_config.json
    ├── adapter_model.bin
    ├── config.json
    ├── tokenizer_config.json
    └── ...
```

### What's Tracked

- **Training Metrics**: Loss, learning rate, gradient norm, etc.
- **System Information**: GPU usage, memory consumption
- **Configuration**: Complete snapshot of all settings used
- **Checkpoints**: Periodic saves for resuming interrupted training
- **Final Model**: Trained LoRA adapters and configuration

### Viewing Metrics

Training metrics are saved in JSON format and can be analyzed with any tool:

```python
import json

with open('experiments/20241205_143022_my-experiment/metrics/training_metrics.json') as f:
    metrics = json.load(f)
    print(f"Final loss: {metrics['train_loss'][-1]}")
```

## Supported Architectures

This toolkit supports most HuggingFace causal language models, including:

- **Falcon** (falcon-7b, falcon-40b)
- **LLaMA** (LLaMA-2, LLaMA-3)
- **Mistral** (mistral-7b, mixtral-8x7b)
- **GPT-2** (gpt2, gpt2-medium, gpt2-large)
- **GPT-Neo** (gpt-neo-125m, gpt-neo-1.3b, gpt-neo-2.7b)
- **OPT** (opt-125m through opt-66b)

### Architecture-Specific Settings

Different model architectures require different `target_modules` for LoRA:

**Falcon:**
```yaml
lora:
  target_modules: ["query_key_value"]
```

**LLaMA/Mistral:**
```yaml
lora:
  target_modules: ["q_proj", "v_proj", "k_proj", "o_proj"]
```

**GPT-2/GPT-Neo:**
```yaml
lora:
  target_modules: ["c_attn", "c_proj"]
```

Refer to `configs/example_llama.yaml` and `config_schema.yaml` for architecture-specific examples.

## Troubleshooting

### Out of Memory (OOM) Errors

If you encounter CUDA out-of-memory errors:

1. **Reduce batch size**:
   ```yaml
   training:
     per_device_batch_size: 2  # Try 1 if still failing
   ```

2. **Increase gradient accumulation**:
   ```yaml
   training:
     gradient_accumulation_steps: 8  # Maintain effective batch size
   ```

3. **Reduce sequence length**:
   ```yaml
   dataset:
     max_length: 256  # Or 128 for very limited memory
   ```

4. **Use a smaller model**: Try a 3B or 1B parameter model instead of 7B

### HuggingFace Hub Authentication

To push models to HuggingFace Hub:

1. **Install HuggingFace CLI**:
   ```bash
   pip install huggingface-hub
   ```

2. **Login**:
   ```bash
   huggingface-cli login
   ```

3. **Set hub_repo in config**:
   ```yaml
   output:
     hub_repo: "your-username/your-model-name"
   ```

### Dataset Format Issues

If your dataset doesn't match the expected format:

1. **Check dataset structure**:
   ```python
   from datasets import load_dataset
   dataset = load_dataset("your-dataset-name")
   print(dataset["train"][0])  # View first example
   ```

2. **Adjust prompt template** to match your column names:
   ```yaml
   dataset:
     prompt_template:
       format: "Question: {question}\nAnswer: {answer}"
       context_column: null  # If no context column exists
   ```

### CUDA Not Available

If `python cli.py info` shows CUDA as unavailable:

1. **Verify NVIDIA driver**:
   ```bash
   nvidia-smi
   ```

2. **Reinstall PyTorch with CUDA**:
   ```bash
   pip install torch --index-url https://download.pytorch.org/whl/cu118
   ```

3. **Check CUDA version compatibility** between your driver and PyTorch

### Training Stalls or Hangs

If training appears to freeze:

1. **Check GPU utilization**:
   ```bash
   watch -n 1 nvidia-smi
   ```

2. **Review logs** for error messages:
   ```bash
   tail -f experiments/*/logs/training.log
   ```

3. **Verify dataset loads correctly** - try with a smaller subset first

## Legacy Scripts

For backwards compatibility, the original command-line scripts are still available:

### train_model.py

Legacy training script with command-line arguments.

```bash
python train_model.py <base-model> <dataset> <output-name> [hf-repo]
```

**Examples:**
```bash
# With HuggingFace Hub push
python train_model.py vilsonrodrigues/falcon-7b-instruct-sharded Amod/mental_health_counseling_conversations finetuned-model akumar23/mental-falcon-7b

# Local only (no Hub push)
python train_model.py vilsonrodrigues/falcon-7b-instruct-sharded Amod/mental_health_counseling_conversations finetuned-model
```

**Note**: The legacy script uses hardcoded parameters. For full control, use the YAML configuration system with `cli.py train`.

### config_gen.py

Legacy utility for generating config.json files.

```bash
python config_gen.py <huggingface-repo>
```

**Example:**
```bash
python config_gen.py akumar23/mental-falcon-7b
```

**Note**: Use `python cli.py generate-config` for the same functionality with better error handling.

## Technical Details

### Quantization Configuration

The toolkit uses bitsandbytes for 4-bit quantization with the following settings:

- **Quantization Type**: NF4 (Normal Float 4-bit)
- **Double Quantization**: Enabled (quantizes quantization constants)
- **Compute Dtype**: float16 or bfloat16
- **Device Map**: Automatic distribution across available GPUs

### LoRA Configuration

Default LoRA parameters optimized for quality and efficiency:

- **Rank (r)**: 16 (controls adapter size)
- **Alpha**: 32 (scaling factor)
- **Dropout**: 0.05 (regularization)
- **Target Modules**: Architecture-specific attention layers
- **Task Type**: CAUSAL_LM (causal language modeling)

### Training Configuration

- **Optimizer**: paged_adamw_8bit (8-bit Adam with paging)
- **Learning Rate Scheduler**: Cosine with warmup
- **Gradient Clipping**: Max norm of 1.0
- **Mixed Precision**: Automatic via bitsandbytes
- **Gradient Checkpointing**: Enabled for memory efficiency

### Dataset Processing

The toolkit expects datasets with conversation or instruction-following format:

- **Prompt Template**: Customizable template for formatting inputs
- **Context Support**: Optional context column for multi-turn conversations
- **Tokenization**: Automatic padding and truncation to max_length
- **Caching**: Tokenized datasets are cached for faster subsequent runs

### System Requirements

- **Python Packages**: See requirements.txt for complete list
  - torch (PyTorch with CUDA support)
  - transformers (HuggingFace model library)
  - datasets (HuggingFace dataset library)
  - peft (Parameter-Efficient Fine-Tuning)
  - bitsandbytes (Quantization library)
  - accelerate (Distributed training support)
  - pyyaml (Configuration file parsing)

- **Disk Space**:
  - Model weights: 5-15GB per 7B model (depending on quantization)
  - Checkpoints: ~2GB per checkpoint
  - Datasets: Varies by dataset size

---

## Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests.

## License

See LICENSE file for details.

## Acknowledgments

This toolkit builds on:
- [HuggingFace Transformers](https://github.com/huggingface/transformers)
- [PEFT](https://github.com/huggingface/peft)
- [bitsandbytes](https://github.com/TimDettmers/bitsandbytes)
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
