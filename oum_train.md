# Training Methods

(training-methods)=

## Introduction

Oumi supports several training methods to accommodate different use cases.

Here's a quick comparison:

| Method | Use Case | Data Required | Compute | Key Features |
|--------|----------|---------------|---------|--------------|
| [Supervised Fine-Tuning (SFT)](#supervised-fine-tuning-sft) | Task adaptation | Input-output pairs | Low | Fine-tunes pre-trained models on specific tasks by providing labeled conversations. |
| [Vision-Language SFT](#vision-language-sft) | Multimodal tasks | Image-text pairs | Moderate | Extends SFT to handle both images and text, enabling image understanding problems. |
| [Pretraining](#pretraining) | Domain adaptation | Raw text | Very High | Trains a language model from scratch or adapts it to a new domain using large amounts of unlabeled text. |
| [Direct Preference Optimization (DPO)](#direct-preference-optimization-dpo) | Preference learning | Preference pairs | Low | Trains a model to align with human preferences by providing pairs of preferred and rejected outputs. |
| [Group Relative Policy Optimization (GRPO)](#group-relative-policy-optimization-grpo) | Reasoning | Input-output pairs | Moderate | Trains a model to improve reasoning skills by providing training examples with concrete answers. |

```{tip}
Oumi supports GRPO on Vision-Language Models with the `VERL_GRPO` trainer.
```

(supervised-fine-tuning-sft)=

## Supervised Fine-Tuning (SFT)

### Overview

Supervised Fine-Tuning (SFT) is the most common approach for adapting a pre-trained language model to specific downstream tasks. This involves fine-tuning the model's parameters on a labeled dataset of input-output pairs, effectively teaching the model to perform the desired task. SFT is effective for a wide range of tasks, including:

- **Question answering:** Answering questions based on given context or knowledge. This could be used to build chatbots that can answer questions about a specific domain or provide general knowledge.
- **Agent development:** Training language models to act as agents that can interact with their environment and perform tasks autonomously. This involves fine-tuning the model on data that demonstrates how to complete tasks, communicate effectively, and make decisions.
- **Tool use:** Fine-tuning models to effectively use external tools (e.g., calculators, APIs, databases) to augment their capabilities. This involves training on data that shows how to call tools, interpret their outputs, and integrate them into problem-solving.
- **Structured data extraction:** Training models to extract structured information from unstructured text. This can be used to extract entities, relationships, or key events from documents, enabling automated data analysis and knowledge base construction.
- **Text generation:** Generating coherent text, code, scripts, email, etc. based on a prompt.

### Data Format

SFT uses the {class}`~oumi.core.types.conversation.Conversation` format, which represents a conversation between a user and an assistant. Each turn in the conversation is represented by a message with a role ("user" or "assistant") and content.

```python
{
    "messages": [
        {
            "role": "user",
            "content": "What is machine learning?"
        },
        {
            "role": "assistant",
            "content": "Machine learning is a type of artificial intelligence that allows software applications to become more accurate in predicting outcomes without being explicitly programmed to do so. Machine learning algorithms use historical data as input to predict new output values."
        }
    ]
}
```

See {doc}`/resources/datasets/sft_datasets` for available SFT datasets.

### Configuration

The `data` section in the configuration file specifies the dataset to use for training. The `training` section defines various training parameters.

```yaml
data:
  train:
    datasets:
      - dataset_name: "text_sft"
        dataset_path: "/path/to/data"
        split: "train"
    collator_name: "text_with_padding"

training:
  trainer_type: "TRL_SFT"
```

See the {gh}`🔧 Model Finetuning Guide <notebooks/Oumi - Finetuning Tutorial.ipynb>` notebook for a complete example.

(vision-language-sft)=

## Vision-Language SFT

### Overview

Vision-Language SFT extends the concept of Supervised Fine-Tuning to handle both images and text. This enables the model to understand and reason about visual information, opening up a wide range of multimodal applications:

- **Image-based instruction following:** Following instructions that involve both text and images. For example, the model could be instructed to analyze and generate a report based on an image of a table.
- **Multimodal Agent Development:** Training agents that can perceive and act in the real world through vision and language. This could include tasks like navigating a physical space, interacting with objects, or following complex instructions.
- **Structured Data Extraction from Images:** Extracting structured data from images, such as tables, forms, or diagrams. This could be used to automate data entry or to extract information from scanned documents.

### Data Format

Vision-Language SFT uses the {class}`~oumi.core.types.conversation.Conversation` format with additional support for images. The `image` field contains the path to the image file.

::::{tab-set-code}
:::{code-block} JSON

{
  "messages": [
    {
      "role": "user",
      "content": [
        {
          "type": "image_url",
          "content": "https://oumi.ai/the_great_wave_off_kanagawa.jpg"
        },
        {
          "type": "text",
          "content": "What is in this image?"
        }
      ]
    },
    {
      "role": "assistant",
      "content": "The image is a traditional Japanese ukiyo-e print."
    }
  ]
}
:::

:::{code-block} python

from oumi.core.types.conversation import Conversation, ContentItem, Message, Role, Type

Conversation(
    messages=[
        Message(
            role=Role.USER,
            content=[
                ContentItem(
                    type=Type.IMAGE_URL,
                    content="https://oumi.ai/the_great_wave_off_kanagawa.jpg"
                ),
                ContentItem(type=Type.TEXT, content="What is in this image?"),
            ],
        ),
        Message(
            role=Role.ASSISTANT,
            content="The image is a traditional Japanese ukiyo-e print."
        )
    ]
)
:::
::::

See {doc}`/resources/datasets/vl_sft_datasets` for available vision-language datasets.

### Configuration

The configuration for Vision-Language SFT is similar to SFT, but with additional parameters for handling images.

```yaml
model:
  model_name: "meta-llama/Llama-3.2-11B-Vision-Instruct"
  chat_template: "llama3-instruct"
  freeze_layers: ["vision_encoder"]  # Freeze specific layers to only train the language model, and not the vision encoder

data:
  train:
    collator_name: "vision_language_with_padding" # Visual features collator
    collator_kwargs: {}  # Optional: Additional collator parameters
    datasets:
      - dataset_name: "vl_sft_jsonl"
        dataset_path: "/path/to/data"
        trust_remote_code: False # Set to true if needed for model-specific processors
        dataset_kwargs:
          processor_name: "meta-llama/Llama-3.2-11B-Vision-Instruct" # Feature generator

training:
  trainer_type: "TRL_SFT"
```

**Note:** You can use `collator_kwargs` to customize the vision-language collator behavior. See the {doc}`configuration guide </user_guides/train/configuration>` for more details and examples.

See the {gh}`🖼️ Oumi Multimodal <notebooks/Oumi - Vision Language Models.ipynb>` notebook for a complete example.

(pretraining)=

## Pretraining

### Overview

The most common pretraining method is Causal Language Modeling (CLM), where the model predicts the next token in a sequence, given the preceding tokens.
Pretraining is the process of training a language model from scratch, or continuing training on a pre-trained model, using large amounts of unlabeled text data. This is a computationally expensive process, but it can result in models with strong general language understanding capabilities.

Pretraining is typically used for:

- **Training models from scratch:** This involves training a new language model from scratch on a massive text corpus.
- **Continuing training on new data:** This involves taking a pre-trained model and continuing to train it on additional data, to improve its performance on existing tasks.
- **Domain adaptation:** This involves adapting a pre-trained model to a specific domain, such as scientific literature or legal documents.

### Data Format

Pretraining uses the {class}`~oumi.core.datasets.BasePretrainingDataset` format, which simply contains the text to be used for training.

```python
{
    "text": "Document text for pretraining..."
}
```

See {doc}`/resources/datasets/pretraining_datasets` section on pretraining datasets.

### Configuration

The configuration for pretraining specifies the dataset and the pretraining approach to use.

```yaml
data:
  train:
    datasets:
      - dataset_name: "text"
        dataset_path: "/path/to/corpus"
        streaming: true  # Stream data from disk and/or network
    pack: true  # Pack multiple documents into a single sequence
    max_length: 2048  # Maximum sequence length

training:
  trainer_type: "OUMI"
```

**Explanation of Configuration Parameters:**

- `streaming`: If set to `true`, the data will be streamed from disk, which is useful for large datasets that don't fit in memory.
- `pack`: If set to `true`, multiple documents will be packed into a single sequence, which can improve efficiency.

(direct-preference-optimization-dpo)=

## Direct Preference Optimization (DPO)

### Overview

Direct Preference Optimization (DPO) is a technique for training language models to align with human preferences. It involves presenting the model with pairs of outputs (e.g., two different responses to the same prompt) and training it to prefer the output that humans prefer. DPO offers several advantages:

- **Training with human preferences:** DPO allows you to directly incorporate human feedback into the training process, leading to models that generate more desirable outputs.
- **Improving output quality without reward models:** Unlike reinforcement learning methods, DPO doesn't require a separate reward model to evaluate the quality of outputs.

### Data Format

DPO uses the {class}`~oumi.core.datasets.BaseDpoDataset` format, which includes the prompt, the chosen output, and the rejected output.

```python
{
    "messages": [
        {
            "role": "user",
            "content": "Write a story about a robot"
        }
    ],
    "chosen": {
        "messages": [
            {
                "role": "assistant",
                "content": "In the year 2045, a robot named..."
            }
        ]
    },
    "rejected": {
        "messages": [
            {
                "role": "assistant",
                "content": "There was this robot who..."
            }
        ]
    }
}
```

See {doc}`/resources/datasets/preference_datasets` section on preference datasets.

### Configuration

The configuration for DPO specifies the training parameters and the DPO settings.

```yaml
data:
  train:
    datasets:
      - dataset_name: "preference_pairs_jsonl"
        dataset_path: "/path/to/data"
    collator_name: "dpo_with_padding"

training:
  trainer_type: "TRL_DPO"  # Use the TRL DPO trainer
```

(group-relative-policy-optimization-grpo)=

## Group Relative Policy Optimization (GRPO)

### Overview

Group Relative Policy Optimization (GRPO) is a technique for training language models using reinforcement learning. A common usage is for training reasoning models on verifiable rewards, i.e. rewards calculated by functions as opposed to a reward model. An example of this is math problems, where there is a correct answer, and correctly-formatted incorrect answers can be given partial credit. While GRPO can be used with reward models, we primarily consider the case of using reward functions here.

Some advantages of GRPO include:

- **No value model:** Unlike PPO, where a value aka critic model has to be trained alongside the actor model to estimate long-term reward, GRPO estimates the baseline from group scores, obviating the need for this model. This reduces training complexity and memory usage.
- **Training on verifiable rewards:** By having reward functions, a separate reward model doesn't have to be trained, reducing complexity and memory usage.
- **Does not require labeled preference data:** Unlike other algorithms like DPO, GRPO doesn't require labeled pairwise preference data. Instead, advantages are calculated by comparing multiple generations for a single prompt.

### Data Format

GRPO datasets should inherit from the {class}`~oumi.core.datasets.BaseExperimentalGrpoDataset` dataset class. Inside this class, you can implement any custom transformation logic you need.

#### TRL_GRPO

For the `TRL_GRPO` trainer, the only requirement is the dataset includes a `"prompt"` column containing either the plaintext prompt, or messages in [conversational format](https://huggingface.co/docs/trl/main/en/dataset_formats#conversational). The other fields, such as metadata, are optional, but are passed into the custom reward function if present. The following is a single example for {class}`~oumi.datasets.grpo.LetterCountGrpoDataset`, which has prompts asking models to count letters in words:

```python
{
    "prompt": [
        {
            "content": 'Your final answer should be an integer written as digits and formatted as "\\boxed{your_answer}". For example, if the answer is 42, you should output "\\boxed{42}".',
            "role": "system",
        },
        {
            "content": "Could you determine the count of 'l's in 'substantial'?",
            "role": "user",
        },
    ],
    "letter_count": 1,
}
```

#### VERL_GRPO

The `VERL_GRPO` trainer has a specific format required for its input dataset. Read their [documentation](https://verl.readthedocs.io/en/latest/preparation/prepare_data.html) for more information. An example for the [Countdown dataset](https://huggingface.co/datasets/Jiayi-Pan/Countdown-Tasks-3to4) is shown below:

```python
{
    "ability": "math",
    "data_source": "countdown",
    "extra_info": {"split": "train"},
    "prompt": [
        {
            "content": "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer.\nUser: Using the numbers [79, 8], create an equation that equals 87. You can use basic arithmetic operations (+, -, *, /) and each number can only be used once. Show your work in <think> </think> tags. And return the final answer in <answer> </answer> tags, for example <answer> (1 + 2) / 3 </answer>.\nAssistant: Let me solve this step by step.\n<think>",
            "role": "user",
        }
    ],
    "reward_model": {
        "ground_truth": {"numbers": [79, 8], "target": 87},
        "style": "rule",
    },
}
```

```{tip}
verl requires paths to Parquet files for the training and validation data. Oumi allows you to use HuggingFace Datasets instead by automatically creating the necessary Parquet files before training.
```

### Reward function

Instead of training a separate reward model which estimates the reward value of a completion, it is common to use reward functions instead. Both the trl and verl frameworks have specific interfaces required for the reward functions used. These are documented in the [trl documentation](https://huggingface.co/docs/trl/main/en/grpo_trainer#using-a-custom-reward-function) and [verl documentation](https://verl.readthedocs.io/en/latest/preparation/reward_function.html) respectively.

### Configuration

#### TRL_GRPO

Configuring the `TRL_GRPO` trainer is similar to most other trl-based trainers in Oumi, like `TRL_SFT`. Most Oumi config fields will be used, as trl's [GRPO config](https://huggingface.co/docs/trl/main/en/grpo_trainer#trl.GRPOConfig) is built on top of HF's config. The following configuration highlights some relevant fields for GRPO:

```yaml
model:
  model_name: "Qwen/Qwen2-0.5B-Instruct"

data:
  train:
    datasets:
      - dataset_name: "trl-lib/tldr"
        split: "train"

training:
  trainer_type: "TRL_GRPO"

  # Specifies the name of a reward function in our reward function registry.
  reward_functions: ["soft_20tokens_completions"]

  grpo:
    use_vllm: True
```

#### VERL_GRPO

verl is an RL training framework created by Alibaba. Many Oumi config fields, which generally correspond to HF config fields, thus are not consumed by verl. The following table shows all Oumi config fields used by the verl trainer, and what fields they map to. An overview of fields in the verl config can be found in their [documentation](https://verl.readthedocs.io/en/latest/examples/config.html).

| Oumi                                            | verl                                                  |
|-------------------------------------------------|-------------------------------------------------------|
| model.model_name                                | actor_rollout_ref.model.path                          |
| data.train.datasets                             | data.train_files                                      |
| data.validation.datasets                        | data.val_files                                        |
| training.grpo.max_completion_length             | data.max_response_length                              |
| training.grpo.use_vllm                          | actor_rollout_ref.rollout.name                        |
| training.grpo.temperature                       | actor_rollout_ref.rollout.temperature                 |
| training.grpo.vllm_gpu_memory_utilization       | actor_rollout_ref.rollout.gpu_memory_utilization      |
| training.enable_gradient_checkpointing          | actor_rollout_ref.model.enable_gradient_checkpointing |
| training.learning_rate                          | actor_rollout_ref.actor.optim.lr                      |
| training.num_train_epochs                       | trainer.total_epochs                                  |
| training.max_steps                              | trainer.total_training_steps                          |
| training.eval_steps/training.eval_strategy      | trainer.test_freq                                     |
| training.save_steps/training.save_epoch         | trainer.save_freq                                     |
| training.resume_from_checkpoint                 | trainer.resume_mode/trainer.resume_from_path          |
| training.try_resume_from_last_checkpoint        | trainer.resume_mode                                   |
| training.logging_strategy/training.enable_wandb | trainer.logger                                        |
| training.run_name                               | trainer.experiment_name                               |
| training.output_dir                             | trainer.default_local_dir                             |

```{tip}
The `training.verl_config_overrides` field can be used to specify any field in the verl config. The values specified in this field will override any values set by the Oumi -> verl mapping above. For example, if you already have your own training/validation Parquet files you want to use, you can directly set `data.train_files` in the override.
```

The following shows a bare-bones Oumi `VERL_GRPO` config.

```yaml
model:
  model_name: "Qwen/Qwen2-0.5B-Instruct"

data:
  train:
    datasets:
      - dataset_name: "Jiayi-Pan/Countdown-Tasks-3to4"
        split: "train"
  # verl requires a validation set.
  validation:
    datasets:
      - dataset_name: "Jiayi-Pan/Countdown-Tasks-3to4"
        split: "test"

training:
  trainer_type: "VERL_GRPO"
  reward_functions: ["countdown"]

  grpo:
    use_vllm: True

  verl_config_overrides:
    # This sets `data.train_batch_size` to 128 in the verl config.
    data:
      train_batch_size: 128
```

## Next Steps

- Explore {doc}`configuration options </user_guides/train/configuration>`
- Set up {doc}`monitoring tools </user_guides/train/monitoring>`

# Training Configuration

## Introduction

This guide covers the configuration options available for training in Oumi. The configuration system is designed to be:

- **Modular**: Each aspect of training (model, data, optimization, etc.) is configured separately
- **Type-safe**: All configuration options are validated at runtime
- **Flexible**: Supports various training scenarios from single-GPU to distributed training
- **Extensible**: Easy to add new configuration options and validate them

The configuration system is built on the {py:obj}`~oumi.core.configs.training_config.TrainingConfig` class, which contains all training settings. This class is composed of several parameter classes:

- [Model Configuration](#model-configuration): Model architecture and loading settings
- [Data Configuration](#data-configuration): Dataset and data loading configuration
- [Training Configuration](#training-configuration): Core training parameters
- [PEFT Configuration](#peft-configuration): Parameter-efficient fine-tuning options
- [FSDP Configuration](#fsdp-configuration): Distributed training settings

All configuration files in Oumi are YAML files, which provide a human-readable format for specifying training settings. The configuration system automatically validates these files and converts them to the appropriate Python objects.

## Basic Structure

A typical configuration file has this structure:

```yaml
model:  # Model settings
  model_name: "HuggingFaceTB/SmolLM2-135M-Instruct"
  trust_remote_code: true

data:   # Dataset settings
  train:
    datasets:
      - dataset_name: "your_dataset"
        split: "train"

training:  # Training parameters
  output_dir: "output/my_run"
  num_train_epochs: 3
  learning_rate: 5e-5
  grpo:   # Optional GRPO settings
    num_generations: 2

peft:  # Optional PEFT settings
  peft_method: "lora"
  lora_r: 8

fsdp:  # Optional FSDP settings
  enable_fsdp: false
```

Each section in the configuration file maps to a specific parameter class and contains settings relevant to that aspect of training. The following sections detail each configuration component.

## Configuration Components

### Model Configuration

Configure the model architecture and loading using the {py:obj}`~oumi.core.configs.params.model_params.ModelParams` class:

```yaml
model:
  # Required
  model_name: "meta-llama/Llama-3.1-8B-Instruct"    # Model ID or path (REQUIRED)

  # Model loading
  adapter_model: null                                # Path to adapter model (auto-detected if model_name is adapter)
  tokenizer_name: null                               # Custom tokenizer name/path (defaults to model_name)
  tokenizer_pad_token: null                          # Override pad token
  tokenizer_kwargs: {}                               # Additional tokenizer args
  model_max_length: null                             # Max sequence length (positive int or null)
  load_pretrained_weights: true                      # Load pretrained weights
  trust_remote_code: false                           # Allow remote code execution (use with trusted models only)
  model_revision: null                               # Model revision to use (e.g., "prequantized")

  # Model precision and hardware
  torch_dtype_str: "float32"                         # Model precision (float32/float16/bfloat16/float64)
  device_map: "auto"                                 # Device placement strategy (auto/null)
  compile: false                                     # JIT compile model (use TrainingParams.compile for training)

  # Attention and optimization
  attn_implementation: null                          # Attention impl (null/sdpa/flash_attention_2/eager)
  enable_liger_kernel: false                         # Enable Liger CUDA kernel for potential speedup

  # Model behavior
  chat_template: null                                # Chat formatting template
  freeze_layers: []                                  # Layer names to freeze during training

  # Additional settings
  model_kwargs: {}                                   # Additional model constructor args
```

### Data Configuration

Configure datasets and data loading using the {py:obj}`~oumi.core.configs.params.data_params.DataParams` class. Each split (`train`/`validation`/`test`) is configured using {py:obj}`~oumi.core.configs.params.data_params.DatasetSplitParams`, and individual datasets are configured using {py:obj}`~oumi.core.configs.params.data_params.DatasetParams`:

```yaml
data:
  train:  # Training dataset configuration
    datasets:  # List of datasets for this split
      - dataset_name: "text_sft"            # Required: Dataset format/type
        dataset_path: "/path/to/data"       # Optional: Path for local datasets
        subset: null                        # Optional: Dataset subset name
        split: "train"                      # Dataset split (default: "train")
        sample_count: null                  # Optional: Number of examples to sample
        mixture_proportion: null            # Optional: Proportion in mixture (0-1)
        shuffle: false                      # Whether to shuffle before sampling
        seed: null                          # Random seed for shuffling
        shuffle_buffer_size: 1000           # Size of shuffle buffer
        trust_remote_code: false            # Trust remote code when loading
        transform_num_workers: null         # Workers for dataset processing
        dataset_kwargs: {}                  # Additional dataset constructor args

    # Split-level settings
    collator_name: "text_with_padding"      # Data collator type
    collator_kwargs: {}                     # Additional collator constructor args
    pack: false                             # Pack text into constant-length chunks
    stream: false                           # Enable dataset streaming
    mixture_strategy: "first_exhausted"     # Strategy for mixing datasets
    seed: null                              # Random seed for mixing
    use_torchdata: false                    # Use `torchdata` (experimental)

  validation:  # Optional validation dataset config
    datasets:
      - dataset_name: "text_sft"
        dataset_path: "/path/to/val"
        split: "validation"
```

Notes:

- When using multiple datasets in a split with `mixture_proportion`:
  - All datasets must specify a `mixture_proportion`
  - The sum of all proportions must equal 1.0
  - The `mixture_strategy` determines how datasets are combined:
    - `first_exhausted`: Stops when any dataset is exhausted
    - `all_exhausted`: Continues until all datasets are exhausted (may oversample)
- When `pack` is enabled:
  - `stream` must also be enabled
- All splits must use the same collator type if specified
- If a collator is specified for validation/test, it must also be specified for train
- `collator_kwargs` allows customizing collator behavior with additional parameters:
  - For `text_with_padding`: Can set `max_variable_sized_dims` to control padding dimensions
  - For `vision_language_with_padding`: Can override `allow_multi_image_inputs` or `main_image_feature`
  - For `vision_language_sft`: Can override `allow_multi_image_inputs`, `truncation_side`, etc.
  - Config-provided kwargs take precedence over automatically determined values

### Training Configuration

Configure the training process using the {py:obj}`~oumi.core.configs.params.training_params.TrainingParams` class:

```yaml
training:
  # Basic settings
  output_dir: "output"                    # Directory for saving outputs
  run_name: null                          # Unique identifier for the run
  seed: 42                                # Random seed for reproducibility
  use_deterministic: false                # Use deterministic CuDNN algorithms

  # Training duration
  num_train_epochs: 3                     # Number of training epochs
  max_steps: -1                           # Max training steps (-1 to use epochs)

  # Batch size settings
  per_device_train_batch_size: 8          # Training batch size per device
  per_device_eval_batch_size: 8           # Evaluation batch size per device
  gradient_accumulation_steps: 1          # Steps before weight update

  # Optimization
  learning_rate: 5e-5                     # Initial learning rate
  optimizer: "adamw_torch"                # Optimizer type ("adam", "adamw", "adamw_torch", "adamw_torch_fused", "sgd", "adafactor")
                                          # "adamw_8bit", "paged_adamw_8bit", "paged_adamw", "paged_adamw_32bit" (requires bitsandbytes)
  weight_decay: 0.0                       # Weight decay for regularization
  max_grad_norm: 1.0                      # Max gradient norm for clipping

  # Optimizer specific settings
  adam_beta1: 0.9                         # Adam beta1 parameter
  adam_beta2: 0.999                       # Adam beta2 parameter
  adam_epsilon: 1e-8                      # Adam epsilon parameter
  sgd_momentum: 0.0                       # SGD momentum (if using SGD)

  # Learning rate schedule
  lr_scheduler_type: "linear"             # LR scheduler type
  warmup_ratio: null                      # Warmup ratio of total steps
  warmup_steps: null                      # Number of warmup steps

  # Mixed precision and performance
  mixed_precision_dtype: "none"           # Mixed precision type ("none", "fp16", "bf16")
  compile: false                          # Whether to JIT compile model
  enable_gradient_checkpointing: false    # Trade compute for memory

  # Checkpointing
  save_steps: 500                         # Save every N steps
  save_epoch: false                       # Save at end of each epoch
  save_final_model: true                  # Save model at end of training
  resume_from_checkpoint: null            # Path to resume from
  try_resume_from_last_checkpoint: false  # Try auto-resume from last checkpoint

  # Evaluation
  eval_strategy: "steps"                  # When to evaluate ("no", "steps", "epoch")
  eval_steps: 500                         # Evaluate every N steps
  metrics_function: null                  # Name of metrics function to use

  # Logging
  log_level: "info"                       # Main logger level
  dep_log_level: "warning"                # Dependencies logger level
  enable_wandb: false                     # Enable Weights & Biases logging
  enable_tensorboard: true                # Enable TensorBoard logging
  logging_strategy: "steps"               # When to log ("steps", "epoch", "no")
  logging_steps: 50                       # Log every N steps
  logging_first_step: false               # Log first step metrics

  # DataLoader settings
  dataloader_num_workers: 0               # Number of dataloader workers (int or "auto")
  dataloader_prefetch_factor: null        # Batches to prefetch per worker (requires workers > 0)
  dataloader_main_process_only: null      # Iterate dataloader on main process only (auto if null)

  # Distributed training
  ddp_find_unused_parameters: false       # Find unused parameters in DDP
  nccl_default_timeout_minutes: null      # NCCL timeout in minutes

  # Performance monitoring
  include_performance_metrics: false      # Include token statistics
  include_alternative_mfu_metrics: false  # Include alternative MFU metrics
  log_model_summary: false                # Print model layer summary
  empty_device_cache_steps: null          # Steps between cache clearing

  # Settings if using GRPO. See below for more details.
  grpo:
    num_generations: null
```

### GRPO Configuration

Configure group relative policy optimization using the {py:obj}`~oumi.core.configs.params.grpo_params.GrpoParams` class:

```yaml
training:
  grpo:
    model_init_kwargs: {}                     # Keyword args for AutoModelForCausalLM.from_pretrained
    max_prompt_length: null                   # Max prompt length in input
    max_completion_length: null               # Max completion length during generation
    num_generations: null                     # Generations per prompt
    temperature: 0.9                          # Sampling temperature (higher = more random)
    remove_unused_columns: false              # If true, only keep the "prompt" column
    repetition_penalty: 1.0                   # Penalty for token repetition (>1 discourages repetition)

    # vLLM settings for generation
    use_vllm: false                           # Use vLLM for generation
    vllm_mode: null                           # Use server or colocate mode for vLLM
    vllm_gpu_memory_utilization: 0.9          # VRAM fraction for vLLM (0-1)
```

### PEFT Configuration

Configure parameter-efficient fine-tuning using the {py:obj}`~oumi.core.configs.params.peft_params.PeftParams` class:

```yaml
peft:
  # LoRA settings
  lora_r: 8                          # Rank of update matrices
  lora_alpha: 8                      # Scaling factor
  lora_dropout: 0.0                  # Dropout probability
  lora_target_modules: null          # Modules to apply LoRA to
  lora_modules_to_save: null         # Modules to unfreeze and train
  lora_bias: "none"                  # Bias training type
  lora_task_type: "CAUSAL_LM"        # Task type for adaptation
  lora_init_weights: "DEFAULT"       # Initialization of LoRA weights

  # Q-LoRA settings
  q_lora: false                      # Enable quantization
  q_lora_bits: 4                     # Quantization bits
  bnb_4bit_quant_type: "fp4"         # 4-bit quantization type
  use_bnb_nested_quant: false        # Use nested quantization
  bnb_4bit_quant_storage: "uint8"    # Storage type for params
  bnb_4bit_compute_dtype: "float32"  # Compute type for params
  llm_int8_skip_modules: "none"      # A list of modules that we do not want to convert in 8-bit.
```

### FSDP Configuration

Configure fully sharded data parallel training using the {py:obj}`~oumi.core.configs.params.fsdp_params.FSDPParams` class:

```yaml
fsdp:
  enable_fsdp: false                        # Enable FSDP training
  sharding_strategy: "FULL_SHARD"           # How to shard model
  cpu_offload: false                        # Offload to CPU
  mixed_precision: null                     # Mixed precision type
  backward_prefetch: "BACKWARD_PRE"         # When to prefetch params
  forward_prefetch: false                   # Prefetch forward results
  use_orig_params: null                     # Use original module params
  state_dict_type: "FULL_STATE_DICT"        # Checkpoint format

  # Auto wrapping settings
  auto_wrap_policy: "NO_WRAP"               # How to wrap layers
  min_num_params: 100000                    # Min params for wrapping
  transformer_layer_cls: null               # Transformer layer class

  # Other settings
  sync_module_states: true                  # Sync states across processes
```

Notes on FSDP sharding strategies:

- `FULL_SHARD`: Shards model parameters, gradients, and optimizer states. Most memory efficient but may impact performance.
- `SHARD_GRAD_OP`: Shards gradients and optimizer states only. Balances memory and performance.
- `HYBRID_SHARD`: Shards parameters within a node, replicates across nodes.
- `NO_SHARD`: No sharding (use DDP instead).
- `HYBRID_SHARD_ZERO2`: Uses SHARD_GRAD_OP within node, replicates across nodes.

## Example Configurations

You can find these examples and many more in the {doc}`/resources/recipes` section.

We aim to provide a comprehensive (and growing) set of recipes for all the common training scenarios:

### Full Fine-tuning (SFT)

This example shows how to fine-tune a small model ('SmolLM2-135M') without any parameter-efficient methods:

````{dropdown} configs/recipes/smollm/sft/135m/quickstart_train.yaml
```{literalinclude} ../../../configs/recipes/smollm/sft/135m/quickstart_train.yaml
:language: yaml
```
````

### Parameter-Efficient Fine-tuning (LoRA)

This example shows how to fine-tune a large model ('Llama-3.1-70b') using LoRA:

````{dropdown} configs/recipes/llama3_1/sft/70b_lora/train.yaml
```{literalinclude} ../../../configs/recipes/llama3_1/sft/70b_lora/train.yaml
:language: yaml
```
````

### Distributed Training (FSDP)

This example shows how to fine-tune a medium-sized model ('Llama-3.1-8b') using FSDP for distributed training:

````{dropdown} configs/recipes/llama3_1/sft/8b_full/train.yaml
```{literalinclude} ../../../configs/recipes/llama3_1/sft/8b_full/train.yaml
:language: yaml
```
````

### Group Relative Policy Optimization (GRPO)

This example shows how to train a model using the GRPO reinforcement learning algorithm:

````{dropdown} configs/examples/grpo_tldr/train.yaml
```{literalinclude} ../../../configs/examples/grpo_tldr/train.yaml
:language: yaml
```
````

### Vision-Language Fine-tuning

This example shows how to fine-tune a vision-language model ('LLaVA-7B'):

````{dropdown} configs/recipes/vision/llava_7b/sft/train.yaml
```{literalinclude} ../../../configs/recipes/vision/llava_7b/sft/train.yaml
:language: yaml
```
````
# Hyperparameter Tuning

(hyperparameter-tuning)=

## Introduction

Finding the right hyperparameters can make the difference between a mediocre model and state-of-the-art performance. Oumi provides `oumi tune`, a built-in hyperparameter optimization module powered by [Optuna](https://optuna.org/) that makes systematic hyperparameter search effortless.

With `oumi tune`, you can:

- 🔍 **Automatic Search**: Systematically search through hyperparameter spaces using advanced algorithms (TPE, random sampling)
- 🎯 **Multi-Objective Optimization**: Optimize for multiple metrics simultaneously (e.g., minimize loss while maximizing accuracy)
- 📊 **Smart Sampling**: Use log-uniform sampling for learning rates, categorical choices for optimizers, and more
- 💾 **Full Tracking**: Automatically save results, best models, and detailed trial logs
- 🚀 **Easy Integration**: Works seamlessly with all Oumi training workflows

## Installation

To use the hyperparameter tuning feature, install Oumi with the `tune` extra:

```bash
pip install oumi[tune]
```

This installs Optuna and related dependencies needed for hyperparameter tuning.

## Quick Start

### Basic Usage

Create a tuning configuration file (`tune.yaml`):

```yaml
model:
  model_name: "HuggingFaceTB/SmolLM2-135M-Instruct"
  model_max_length: 2048
  torch_dtype_str: "bfloat16"

data:
  train:
    datasets:
      - dataset_name: "yahma/alpaca-cleaned"
        split: "train[90%:]"
  validation:
    datasets:
      - dataset_name: "yahma/alpaca-cleaned"
        split: "train[:10%]"

tuning:
  n_trials: 10

  # Define hyperparameters to search
  tunable_training_params:
    learning_rate:
      type: loguniform
      low: 1e-5
      high: 1e-2

  # Fixed parameters (not tuned)
  fixed_training_params:
    trainer_type: TRL_SFT
    per_device_train_batch_size: 1
    max_steps: 1000

  # Metrics to optimize
  evaluation_metrics: ["eval_loss"]
  evaluation_direction: ["minimize"]

  tuner_type: OPTUNA
  tuner_sampler: "TPESampler"
```

Run tuning with a single command:

```bash
oumi tune -c tune.yaml
```

## Configuration

### Parameter Types

Oumi supports several parameter types for defining search spaces:

#### Categorical Parameters

Choose from a discrete set of options:

```yaml
tunable_training_params:
  optimizer:
    type: categorical
    choices: ["adamw_torch", "sgd", "adafactor"]
```

#### Integer Parameters

Sample integers within a range:

```yaml
tunable_training_params:
  gradient_accumulation_steps:
    type: int
    low: 1
    high: 8
```

#### Float Parameters (Uniform)

Sample floats uniformly within a range:

```yaml
tunable_training_params:
  warmup_ratio:
    type: uniform
    low: 0.0
    high: 0.3
```

#### Float Parameters (Log-Uniform)

Sample floats on a logarithmic scale (ideal for learning rates):

```yaml
tunable_training_params:
  learning_rate:
    type: loguniform
    low: 1e-5
    high: 1e-2
```

### Training Parameters

You can tune any training parameter by adding it to `tunable_training_params`:

```yaml
tuning:
  tunable_training_params:
    learning_rate:
      type: loguniform
      low: 1e-5
      high: 1e-2

    per_device_train_batch_size:
      type: categorical
      choices: [2, 4, 8]

    num_train_epochs:
      type: int
      low: 1
      high: 5

    weight_decay:
      type: uniform
      low: 0.0
      high: 0.1
```

### PEFT Parameters

For efficient fine-tuning with LoRA or QLoRA, tune PEFT parameters:

```yaml
tuning:
  tunable_peft_params:
    lora_r:
      type: categorical
      choices: [4, 8, 16, 32]

    lora_alpha:
      type: categorical
      choices: [8, 16, 32, 64]

    lora_dropout:
      type: uniform
      low: 0.0
      high: 0.1

  fixed_peft_params:
    q_lora: false
    lora_target_modules: ["q_proj", "v_proj"]
```

### Multi-Objective Optimization

Optimize for multiple metrics simultaneously:

```yaml
tuning:
  # Multiple evaluation metrics
  evaluation_metrics: ["eval_loss", "eval_mean_token_accuracy"]
  evaluation_direction: ["minimize", "maximize"]

  # Optuna will find the Pareto frontier of trials
```

When using multi-objective optimization, use `get_best_trials()` (plural) instead of `get_best_trial()` to retrieve the Pareto-optimal trials.

## Tuner Configuration

### Tuner Type

Currently, Oumi supports the Optuna tuner:

```yaml
tuning:
  tuner_type: OPTUNA
```

### Samplers

Choose from different sampling strategies:

#### TPE Sampler (Recommended)

Tree-structured Parzen Estimator - efficient Bayesian optimization:

```yaml
tuning:
  tuner_sampler: "TPESampler"
```

#### Random Sampler

Simple random sampling (good baseline):

```yaml
tuning:
  tuner_sampler: "RandomSampler"
```

## Advanced Usage

### Custom Evaluation Metrics

You can define custom evaluation metrics to optimize:

```yaml
tuning:
  evaluation_metrics: ["eval_loss", "eval_accuracy", "custom_metric"]
  evaluation_direction: ["minimize", "maximize", "maximize"]

  custom_eval_metrics:
    - name: "custom_metric"
      function: "my_module.compute_custom_metric"
```

## Output and Results

### Output Structure

Tuning results are saved in the output directory:

```
tuning_output/
├── trials_results.csv          # Summary of all trials
├── trial_0/                    # First trial
│   ├── checkpoint-100/
│   └── logs/
├── trial_1/                    # Second trial
│   ├── checkpoint-100/
│   └── logs/
└── ...
```

### Results CSV

The `trials_results.csv` file contains:

- Trial number
- Hyperparameter values for each trial
- Evaluation metrics for each trial
- Trial status (completed, failed, etc.)

### Best Model

The best model checkpoint is saved in the trial directory with the best evaluation metric(s).

## See Also

- {doc}`/user_guides/train/train` - Training guide
- {doc}`/resources/recipes` - Pre-configured recipes
- [Optuna Documentation](https://optuna.readthedocs.io/) - Optuna's official documentation
- {doc}`/api/oumi.core.tuners` - API reference for tuners