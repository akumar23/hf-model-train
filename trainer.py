"""
HuggingFace Model Fine-Tuning Trainer Module

This module provides classes for automated model training with:
- YAML configuration loading and validation
- Experiment tracking and logging
- Checkpoint management
- Batch training support
"""

import os
import sys
import json
import logging
import shutil
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, List

import yaml
import torch
import transformers
from datasets import load_dataset
from peft import (
    LoraConfig,
    PeftConfig,
    PeftModel,
    get_peft_model,
    prepare_model_for_kbit_training
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)


# =============================================================================
# CONFIGURATION CLASSES
# =============================================================================

@dataclass
class ModelConfig:
    """Model configuration"""
    base_model: str
    trust_remote_code: bool = True
    device_map: str = "auto"


@dataclass
class QuantizationConfig:
    """Quantization configuration"""
    load_in_4bit: bool = True
    use_double_quant: bool = True
    quant_type: str = "nf4"
    compute_dtype: str = "bfloat16"
    
    def get_torch_dtype(self):
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        return dtype_map.get(self.compute_dtype, torch.bfloat16)


@dataclass
class LoRAConfig:
    """LoRA configuration"""
    r: int = 16
    lora_alpha: int = 32
    target_modules: List[str] = field(default_factory=lambda: ["query_key_value"])
    lora_dropout: float = 0.05
    bias: str = "none"
    task_type: str = "CAUSAL_LM"


@dataclass
class DatasetConfig:
    """Dataset configuration"""
    name: str
    train_split: str = "train"
    val_split: Optional[str] = None
    prompt_template: Dict[str, Any] = field(default_factory=dict)
    max_length: int = 512
    truncation: bool = True
    padding: bool = True


@dataclass
class TrainingConfig:
    """Training configuration"""
    num_epochs: int = 3
    per_device_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.05
    optimizer: str = "paged_adamw_8bit"
    fp16: bool = True
    bf16: bool = False
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    logging_steps: int = 10
    save_steps: int = 100
    save_total_limit: int = 3
    gradient_checkpointing: bool = True
    evaluation_strategy: str = "no"
    eval_steps: Optional[int] = None


@dataclass
class GenerationConfig:
    """Generation configuration"""
    max_new_tokens: int = 200
    temperature: float = 0.7
    top_p: float = 0.7
    top_k: int = 50
    num_return_sequences: int = 1
    do_sample: bool = True


@dataclass
class OutputConfig:
    """Output configuration"""
    model_dir: str = "finetuned-model"
    experiments_dir: str = "experiments"
    experiment_name: str = "experiment"
    hub_repo: str = ""
    generate_config: bool = True
    push_tokenizer: bool = True


@dataclass
class EnvironmentConfig:
    """Environment configuration"""
    cuda_visible_devices: str = "0"
    seed: int = 42
    deterministic: bool = False


@dataclass
class LoggingConfig:
    """Logging configuration"""
    level: str = "INFO"
    log_to_file: bool = True
    log_to_console: bool = True
    verbose: bool = False


@dataclass
class FullConfig:
    """Complete training configuration"""
    model: ModelConfig
    quantization: QuantizationConfig
    lora: LoRAConfig
    dataset: DatasetConfig
    training: TrainingConfig
    generation: GenerationConfig
    output: OutputConfig
    environment: EnvironmentConfig
    logging: LoggingConfig


# =============================================================================
# CONFIGURATION LOADER
# =============================================================================

class ConfigLoader:
    """Loads and validates training configuration from YAML files"""
    
    DEFAULTS = {
        "quantization": {
            "load_in_4bit": True,
            "use_double_quant": True,
            "quant_type": "nf4",
            "compute_dtype": "bfloat16",
        },
        "lora": {
            "r": 16,
            "lora_alpha": 32,
            "target_modules": ["query_key_value"],
            "lora_dropout": 0.05,
            "bias": "none",
            "task_type": "CAUSAL_LM",
        },
        "training": {
            "num_epochs": 3,
            "per_device_batch_size": 1,
            "gradient_accumulation_steps": 4,
            "learning_rate": 2e-4,
            "lr_scheduler_type": "cosine",
            "warmup_ratio": 0.05,
            "optimizer": "paged_adamw_8bit",
            "fp16": True,
            "logging_steps": 10,
            "save_steps": 100,
            "save_total_limit": 3,
            "gradient_checkpointing": True,
            "evaluation_strategy": "no",
        },
        "generation": {
            "max_new_tokens": 200,
            "temperature": 0.7,
            "top_p": 0.7,
            "top_k": 50,
            "num_return_sequences": 1,
            "do_sample": True,
        },
        "output": {
            "model_dir": "finetuned-model",
            "experiments_dir": "experiments",
            "experiment_name": "experiment",
            "hub_repo": "",
            "generate_config": True,
            "push_tokenizer": True,
        },
        "environment": {
            "cuda_visible_devices": "0",
            "seed": 42,
            "deterministic": False,
        },
        "logging": {
            "level": "INFO",
            "log_to_file": True,
            "log_to_console": True,
            "verbose": False,
        },
    }
    
    @classmethod
    def load(cls, config_path: str) -> FullConfig:
        """Load configuration from YAML file"""
        with open(config_path, 'r') as f:
            raw_config = yaml.safe_load(f)
        
        # Validate required fields
        cls._validate_required(raw_config)
        
        # Merge with defaults
        config = cls._merge_with_defaults(raw_config)
        
        # Create config objects
        return FullConfig(
            model=ModelConfig(**config.get("model", {})),
            quantization=QuantizationConfig(**config.get("quantization", {})),
            lora=LoRAConfig(**config.get("lora", {})),
            dataset=DatasetConfig(**config.get("dataset", {})),
            training=TrainingConfig(**config.get("training", {})),
            generation=GenerationConfig(**config.get("generation", {})),
            output=OutputConfig(**config.get("output", {})),
            environment=EnvironmentConfig(**config.get("environment", {})),
            logging=LoggingConfig(**config.get("logging", {})),
        )
    
    @classmethod
    def _validate_required(cls, config: Dict) -> None:
        """Validate required configuration fields"""
        errors = []
        
        if "model" not in config or "base_model" not in config.get("model", {}):
            errors.append("model.base_model is required")
        
        if "dataset" not in config or "name" not in config.get("dataset", {}):
            errors.append("dataset.name is required")
        
        if errors:
            raise ValueError(f"Configuration validation failed:\n" + "\n".join(f"  - {e}" for e in errors))
    
    @classmethod
    def _merge_with_defaults(cls, config: Dict) -> Dict:
        """Merge user config with defaults"""
        merged = {}
        
        for section, defaults in cls.DEFAULTS.items():
            user_section = config.get(section, {})
            merged[section] = {**defaults, **user_section}
        
        # Model section has no defaults
        merged["model"] = config.get("model", {})
        merged["dataset"] = {**config.get("dataset", {})}
        
        return merged
    
    @classmethod
    def validate_config_file(cls, config_path: str) -> List[str]:
        """Validate a config file and return any warnings/errors"""
        issues = []
        
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            return [f"YAML parsing error: {e}"]
        
        # Check required fields
        if "model" not in config:
            issues.append("ERROR: 'model' section is required")
        elif "base_model" not in config.get("model", {}):
            issues.append("ERROR: 'model.base_model' is required")
        
        if "dataset" not in config:
            issues.append("ERROR: 'dataset' section is required")
        elif "name" not in config.get("dataset", {}):
            issues.append("ERROR: 'dataset.name' is required")
        
        # Check for common issues
        lora = config.get("lora", {})
        if lora.get("r", 16) > 64:
            issues.append("WARNING: LoRA r > 64 may use excessive memory")
        
        training = config.get("training", {})
        if training.get("learning_rate", 2e-4) > 1e-3:
            issues.append("WARNING: Learning rate > 1e-3 may cause instability")
        
        return issues


# =============================================================================
# EXPERIMENT TRACKER
# =============================================================================

@dataclass
class TrainingMetrics:
    """Structured training metrics"""
    epoch: int = 0
    step: int = 0
    loss: float = 0.0
    learning_rate: float = 0.0
    train_runtime: float = 0.0
    train_samples_per_second: float = 0.0
    train_steps_per_second: float = 0.0
    total_flos: float = 0.0
    extra: Dict[str, Any] = field(default_factory=dict)


class ExperimentTracker:
    """Tracks experiments with logging and metrics"""
    
    def __init__(self, config: FullConfig, experiment_dir: Optional[str] = None):
        self.config = config
        
        # Create experiment directory
        if experiment_dir:
            self.experiment_dir = Path(experiment_dir)
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            exp_name = config.output.experiment_name
            self.experiment_dir = Path(config.output.experiments_dir) / f"{timestamp}_{exp_name}"
        
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        self.logs_dir = self.experiment_dir / "logs"
        self.metrics_dir = self.experiment_dir / "metrics"
        self.checkpoints_dir = self.experiment_dir / "checkpoints"
        
        for d in [self.logs_dir, self.metrics_dir, self.checkpoints_dir]:
            d.mkdir(exist_ok=True)
        
        # Setup logging
        self._setup_logging()
        
        # Save configuration
        self._save_config()
        
        # Initialize metrics storage
        self.metrics_history: List[TrainingMetrics] = []
    
    def _setup_logging(self):
        """Setup logging handlers"""
        self.logger = logging.getLogger(f"experiment_{self.config.output.experiment_name}")
        self.logger.setLevel(getattr(logging, self.config.logging.level))
        self.logger.handlers = []  # Clear existing handlers
        
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        if self.config.logging.log_to_console:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
        
        if self.config.logging.log_to_file:
            file_handler = logging.FileHandler(self.logs_dir / "training.log")
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
    
    def _save_config(self):
        """Save configuration to experiment directory"""
        config_path = self.experiment_dir / "config.yaml"
        
        # Convert dataclasses to dict
        config_dict = {
            "model": asdict(self.config.model),
            "quantization": asdict(self.config.quantization),
            "lora": asdict(self.config.lora),
            "dataset": asdict(self.config.dataset),
            "training": asdict(self.config.training),
            "generation": asdict(self.config.generation),
            "output": asdict(self.config.output),
            "environment": asdict(self.config.environment),
            "logging": asdict(self.config.logging),
        }
        
        with open(config_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)
        
        self.logger.info(f"Configuration saved to {config_path}")
    
    def log(self, message: str, level: str = "info"):
        """Log a message"""
        getattr(self.logger, level)(message)
    
    def log_metrics(self, metrics: TrainingMetrics):
        """Log training metrics"""
        self.metrics_history.append(metrics)
        self.log(f"Step {metrics.step}: loss={metrics.loss:.4f}, lr={metrics.learning_rate:.2e}")
    
    def save_final_metrics(self):
        """Save all metrics to JSON file"""
        metrics_path = self.metrics_dir / "training_metrics.json"
        
        metrics_data = {
            "experiment_name": self.config.output.experiment_name,
            "timestamp": datetime.now().isoformat(),
            "history": [asdict(m) for m in self.metrics_history],
        }
        
        with open(metrics_path, 'w') as f:
            json.dump(metrics_data, f, indent=2)
        
        self.log(f"Metrics saved to {metrics_path}")
    
    def get_checkpoint_dir(self) -> str:
        """Get checkpoint directory path"""
        return str(self.checkpoints_dir)
    
    def get_model_output_dir(self) -> str:
        """Get model output directory path"""
        return str(self.experiment_dir / self.config.output.model_dir)


# =============================================================================
# MODEL TRAINER
# =============================================================================

class ModelTrainer:
    """Main training class for HuggingFace models with PEFT"""
    
    def __init__(self, config: FullConfig, tracker: Optional[ExperimentTracker] = None):
        self.config = config
        self.tracker = tracker or ExperimentTracker(config)
        
        # Set environment
        os.environ["CUDA_VISIBLE_DEVICES"] = config.environment.cuda_visible_devices
        
        # Set seed
        if config.environment.seed:
            transformers.set_seed(config.environment.seed)
        
        self.model = None
        self.tokenizer = None
        self.trainer = None
    
    def _create_bnb_config(self) -> BitsAndBytesConfig:
        """Create BitsAndBytes quantization config"""
        return BitsAndBytesConfig(
            load_in_4bit=self.config.quantization.load_in_4bit,
            bnb_4bit_use_double_quant=self.config.quantization.use_double_quant,
            bnb_4bit_quant_type=self.config.quantization.quant_type,
            bnb_4bit_compute_dtype=self.config.quantization.get_torch_dtype()
        )
    
    def _create_lora_config(self) -> LoraConfig:
        """Create LoRA configuration"""
        return LoraConfig(
            r=self.config.lora.r,
            lora_alpha=self.config.lora.lora_alpha,
            target_modules=self.config.lora.target_modules,
            lora_dropout=self.config.lora.lora_dropout,
            bias=self.config.lora.bias,
            task_type=self.config.lora.task_type
        )
    
    def load_model(self):
        """Load and prepare model for training"""
        self.tracker.log(f"Loading model: {self.config.model.base_model}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model.base_model)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model with quantization
        bnb_config = self._create_bnb_config()
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model.base_model,
            device_map=self.config.model.device_map,
            trust_remote_code=self.config.model.trust_remote_code,
            quantization_config=bnb_config
        )
        
        # Prepare for training
        if self.config.training.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
        
        self.model = prepare_model_for_kbit_training(self.model)
        
        # Apply LoRA
        lora_config = self._create_lora_config()
        self.model = get_peft_model(self.model, lora_config)
        
        # Log trainable parameters
        trainable, total = self._count_parameters()
        self.tracker.log(f"Trainable parameters: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
        
        # Setup generation config
        self._setup_generation_config()
    
    def _count_parameters(self) -> tuple:
        """Count trainable and total parameters"""
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        return trainable, total
    
    def _setup_generation_config(self):
        """Setup generation configuration"""
        gen_config = self.model.generation_config
        gen_config.max_new_tokens = self.config.generation.max_new_tokens
        gen_config.temperature = self.config.generation.temperature
        gen_config.top_p = self.config.generation.top_p
        gen_config.top_k = self.config.generation.top_k
        gen_config.num_return_sequences = self.config.generation.num_return_sequences
        gen_config.do_sample = self.config.generation.do_sample
        gen_config.pad_token_id = self.tokenizer.eos_token_id
        gen_config.eos_token_id = self.tokenizer.eos_token_id
    
    def load_dataset(self):
        """Load and prepare dataset"""
        self.tracker.log(f"Loading dataset: {self.config.dataset.name}")
        
        # Load dataset
        data = load_dataset(self.config.dataset.name)
        
        # Get context column for encoding
        context_col = self.config.dataset.prompt_template.get("context_column", "Context")
        
        # Tokenize dataset
        def tokenize_function(examples):
            return self.tokenizer(
                examples[context_col],
                truncation=self.config.dataset.truncation,
                padding=self.config.dataset.padding,
                max_length=self.config.dataset.max_length
            )
        
        self.encoded_dataset = data.map(tokenize_function, batched=True)
        
        self.tracker.log(f"Dataset loaded: {len(data[self.config.dataset.train_split])} training samples")
        
        return self.encoded_dataset
    
    def _create_training_args(self, checkpoint_dir: Optional[str] = None) -> TrainingArguments:
        """Create training arguments"""
        output_dir = checkpoint_dir or self.tracker.get_checkpoint_dir()
        
        return TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=self.config.training.per_device_batch_size,
            gradient_accumulation_steps=self.config.training.gradient_accumulation_steps,
            num_train_epochs=self.config.training.num_epochs,
            learning_rate=self.config.training.learning_rate,
            lr_scheduler_type=self.config.training.lr_scheduler_type,
            warmup_ratio=self.config.training.warmup_ratio,
            optim=self.config.training.optimizer,
            fp16=self.config.training.fp16,
            bf16=self.config.training.bf16,
            weight_decay=self.config.training.weight_decay,
            max_grad_norm=self.config.training.max_grad_norm,
            logging_steps=self.config.training.logging_steps,
            save_steps=self.config.training.save_steps,
            save_total_limit=self.config.training.save_total_limit,
            evaluation_strategy=self.config.training.evaluation_strategy,
            remove_unused_columns=False,
            report_to="none",  # Disable default reporters
        )
    
    def train(self, resume_from_checkpoint: Optional[str] = None):
        """Run training"""
        self.tracker.log("Starting training...")
        
        # Create training arguments
        training_args = self._create_training_args()
        
        # Create trainer
        train_split = self.config.dataset.train_split
        self.trainer = Trainer(
            model=self.model,
            train_dataset=self.encoded_dataset[train_split]["input_ids"],
            args=training_args,
            data_collator=DataCollatorForLanguageModeling(self.tokenizer, mlm=False)
        )
        
        # Disable cache for training
        self.model.config.use_cache = False
        
        # Train
        try:
            if resume_from_checkpoint:
                self.tracker.log(f"Resuming from checkpoint: {resume_from_checkpoint}")
                self.trainer.train(resume_from_checkpoint=resume_from_checkpoint)
            else:
                self.trainer.train()
            
            # Log final metrics
            if self.trainer.state.log_history:
                for entry in self.trainer.state.log_history:
                    if "loss" in entry:
                        metrics = TrainingMetrics(
                            step=entry.get("step", 0),
                            loss=entry.get("loss", 0),
                            learning_rate=entry.get("learning_rate", 0),
                        )
                        self.tracker.log_metrics(metrics)
            
            self.tracker.log("Training completed successfully!")
            
        except KeyboardInterrupt:
            self.tracker.log("Training interrupted by user", level="warning")
            raise
        except Exception as e:
            self.tracker.log(f"Training failed: {e}", level="error")
            raise
    
    def save_model(self):
        """Save the trained model"""
        model_dir = self.tracker.get_model_output_dir()
        self.tracker.log(f"Saving model to {model_dir}")
        
        self.model.save_pretrained(model_dir)
        
        if self.config.output.push_tokenizer:
            self.tokenizer.save_pretrained(model_dir)
        
        # Generate config.json if requested
        if self.config.output.generate_config:
            self._generate_config_json(model_dir)
        
        # Push to Hub if configured
        if self.config.output.hub_repo:
            self._push_to_hub()
        
        # Save final metrics
        self.tracker.save_final_metrics()
    
    def _generate_config_json(self, model_dir: str):
        """Generate config.json for the model"""
        self.tracker.log("Generating config.json...")
        
        config_path = Path(model_dir) / "config.json"
        self.model.config.to_json_file(str(config_path))
        
        self.tracker.log(f"config.json saved to {config_path}")
    
    def _push_to_hub(self):
        """Push model to HuggingFace Hub"""
        hub_repo = self.config.output.hub_repo
        self.tracker.log(f"Pushing model to HuggingFace Hub: {hub_repo}")
        
        try:
            self.model.push_to_hub(hub_repo, use_auth_token=True)
            
            if self.config.output.push_tokenizer:
                self.tokenizer.push_to_hub(hub_repo, use_auth_token=True)
            
            self.tracker.log(f"Model pushed to Hub successfully!")
        except Exception as e:
            self.tracker.log(f"Failed to push to Hub: {e}", level="error")
            raise
    
    def run(self, resume_from_checkpoint: Optional[str] = None):
        """Run complete training pipeline"""
        try:
            self.load_model()
            self.load_dataset()
            self.train(resume_from_checkpoint=resume_from_checkpoint)
            self.save_model()
            self.tracker.log("Training pipeline completed!")
        except KeyboardInterrupt:
            self.tracker.log("Pipeline interrupted by user", level="warning")
            sys.exit(1)
        except Exception as e:
            self.tracker.log(f"Pipeline failed: {e}", level="error")
            raise


# =============================================================================
# BATCH TRAINER
# =============================================================================

class BatchTrainer:
    """Runs multiple training configurations sequentially"""
    
    def __init__(self, config_paths: List[str]):
        self.config_paths = config_paths
        self.results = []
    
    def run(self):
        """Run all configurations"""
        print(f"Starting batch training with {len(self.config_paths)} configurations...")
        
        for i, config_path in enumerate(self.config_paths, 1):
            print(f"\n{'='*60}")
            print(f"Training {i}/{len(self.config_paths)}: {config_path}")
            print(f"{'='*60}\n")
            
            try:
                config = ConfigLoader.load(config_path)
                trainer = ModelTrainer(config)
                trainer.run()
                
                self.results.append({
                    "config": config_path,
                    "status": "success",
                    "experiment_dir": str(trainer.tracker.experiment_dir)
                })
                
            except Exception as e:
                print(f"ERROR: Training failed for {config_path}: {e}")
                self.results.append({
                    "config": config_path,
                    "status": "failed",
                    "error": str(e)
                })
        
        # Print summary
        self._print_summary()
    
    def _print_summary(self):
        """Print batch training summary"""
        print(f"\n{'='*60}")
        print("BATCH TRAINING SUMMARY")
        print(f"{'='*60}")
        
        success = sum(1 for r in self.results if r["status"] == "success")
        failed = len(self.results) - success
        
        print(f"Total: {len(self.results)} | Success: {success} | Failed: {failed}")
        print()
        
        for r in self.results:
            status = "OK" if r["status"] == "success" else "FAILED"
            print(f"  [{status}] {r['config']}")
            if r["status"] == "success":
                print(f"         Output: {r['experiment_dir']}")
            else:
                print(f"         Error: {r['error']}")


# =============================================================================
# CONFIG GENERATOR (from trained model)
# =============================================================================

def generate_config_from_model(model_path: str, output_path: str = "config.json"):
    """Generate config.json from a trained PEFT model"""
    
    print(f"Loading PEFT config from {model_path}...")
    
    # Setup quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    # Load PEFT config to get base model
    peft_config = PeftConfig.from_pretrained(model_path)
    
    print(f"Base model: {peft_config.base_model_name_or_path}")
    print("Loading base model for config extraction...")
    
    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        peft_config.base_model_name_or_path,
        return_dict=True,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Save config
    model.config.to_json_file(output_path)
    print(f"Config saved to {output_path}")
    
    return output_path


if __name__ == "__main__":
    # Simple test
    print("Trainer module loaded successfully!")
    print("Use cli.py for command-line interface")
