# Inference Engines

Oumi's inference API provides a unified interface for multiple inference engines through the `InferenceEngine` class.

In this guide, we'll go through each supported engine, what they are best for, and how to get started using them.

## Introduction

Before digging into specific engines, let's look at the basic patterns for initializing both local and remote inference engines.

These patterns will be consistent across all engine types, making it easy to switch between them as your needs change.

**Local Inference**

Let's start with a basic example of how to use the `VLLMInferenceEngine` to run inference on a local model.

```python
from oumi.inference import VLLMInferenceEngine
from oumi.core.configs import ModelParams

# Local inference with vLLM
engine = VLLMInferenceEngine(
    ModelParams(
        model_name="meta-llama/Llama-3.2-1B-Instruct",
    )
)
```

**Using the CLI**

You can also specify configuration in YAML, and use the CLI to run inference:

```bash
oumi infer --engine VLLM --model.model_name meta-llama/Llama-3.2-1B-Instruct
```

Checkout the {doc}`inference_cli` for more information on how to use the CLI.

**Cloud APIs**

Remote inference engines (i.e. API based) require a `RemoteParams` object to be passed in.

The `RemoteParams` object contains the API URL and any necessary API keys. For example, here is to use Claude Sonnet 3.5:

```{testcode}
from oumi.inference import AnthropicInferenceEngine
from oumi.core.configs import ModelParams, RemoteParams

engine = AnthropicInferenceEngine(
    model_params=ModelParams(
        model_name="claude-3-5-sonnet-20240620",
    ),
    remote_params=RemoteParams(
        api_key_env_varname="ANTHROPIC_API_KEY",
    )
)
```

**Supported Parameters**

Each inference engine supports a different set of parameters (for example, different generation parameters, or specific model kwargs).

Make sure to check the {doc}`configuration` for an exhaustive list of supported parameters, and the reference page for the specific engine you are using to find the parameters it supports.

For example, the supported parameters for the `VLLMInferenceEngine` can be found in {py:meth}`~oumi.inference.VLLMInferenceEngine.get_supported_params`.

## Local Inference

This next section covers setting up and optimizing local inference engines for running models directly on your machine, whether you're running on a laptop or a server with multiple GPUs.

Local inference is ideal for running your own fine-tuned models, and in general for development, testing, and scenarios where you need complete control over your inference environment.

### Hardware Recommendations

The following tables provide a rough estimate of the memory requirements for different model sizes using both BF16 and Q4 quantization.

The actual memory requirements might vary based on the specific quantization implementation and additional optimizations used.

Also note that Q4 quantization typically comes with some degradation in model quality, though the impact varies by model architecture and task.

**BF16 / FP16 (16-bit)**

| Model Size | GPU VRAM              | Notes |
|------------|----------------------|--------|
| 1B         | ~2 GB                | Can run on most modern GPUs |
| 3B         | ~6 GB                | Can run on mid-range GPUs |
| 7B         | ~14 GB               | Can run on consumer GPUs like RTX 3090 or RX 7900 XTX |
| 13B        | ~26 GB               | Requires high-end GPU or multiple GPUs |
| 33B        | ~66 GB               | Requires enterprise GPUs or multi-GPU setup |
| 70B        | ~140 GB              | Typically requires multiple A100s or H100s |

**Q4 (4-bit)**

| Model Size | GPU VRAM             | Notes |
|------------|----------------------|--------|
| 1B         | ~0.5 GB              | Can run on most integrated GPUs |
| 3B         | ~1.5 GB              | Can run on entry-level GPUs |
| 7B         | ~3.5 GB              | Can run on most gaming GPUs |
| 13B        | ~6.5 GB              | Can run on mid-range GPUs |
| 33B        | ~16.5 GB             | Can run on high-end consumer GPUs |
| 70B        | ~35 GB               | Can run on professional GPUs |

### vLLM Engine

[vLLM](https://github.com/vllm-project/vllm) is a high-performance inference engine that implements state-of-the-art serving techniques like PagedAttention for optimal memory usage and throughput.

vLLM is our recommended choice for production deployments on GPUs.

**Installation**

First, make sure to install the vLLM package:

```bash
pip install vllm
# Alternatively, install all Oumi GPU dependencies, which takes care of installing a
# vLLM version compatible with your current Oumi version.
pip install oumi[gpu]
```

**Basic Usage**

```python
engine = VLLMInferenceEngine(
    ModelParams(
        model_name="meta-llama/Llama-3.1-8B-Instruct",
    )
)
```

**Tensor Parallel Inference**

For multi-GPU setups, you can leverage tensor parallelism:

```python
# Tensor parallel inference
model_params = ModelParams(
        model_name="meta-llama/Llama-3.2-1B-Instruct",
        model_kwargs={
            "tensor_parallel_size": 2,        # Set to number of GPUs
            "gpu_memory_utilization": 1.0,    # Memory usage
            "enable_prefix_caching": True,    # Enable prefix caching
        }
)
```

**Serving LoRA Adapters**

vLLM supports serving LoRA (Low-Rank Adaptation) adapters, allowing you to use fine-tuned models without loading the full model weights. This is particularly useful when you've fine-tuned a base model and want to serve the adapted version.

To serve a LoRA adapter, specify the `adapter_model` parameter pointing to your LoRA checkpoint:

```python
engine = VLLMInferenceEngine(
    ModelParams(
        model_name="meta-llama/Llama-3.1-8B-Instruct",  # Base model
        adapter_model="path/to/lora/adapter",           # LoRA adapter path
    )
)
```

The LoRA adapter can be:
- A local directory containing the adapter weights
- A HuggingFace Hub model ID (e.g., `username/model-lora-adapter`)

vLLM will automatically:
- Load the base model
- Apply the LoRA adapter weights
- Configure the appropriate LoRA rank from the adapter checkpoint

**Important Notes:**

- Not all model architectures support LoRA adapters in vLLM. Check the [vLLM supported models documentation](https://docs.vllm.ai/en/latest/models/supported_models.html) for compatibility.
- The base model specified in `model_name` must match the base model used during LoRA fine-tuning.
- LoRA serving works with both single-GPU and multi-GPU (tensor parallel) setups.

**Resources**

- [vLLM Documentation](https://vllm.readthedocs.io/en/latest/)
- [vLLM LoRA Support](https://docs.vllm.ai/en/latest/models/lora.html)

### LlamaCPP Engine

For scenarios where GPU resources are limited or unavailable, the [LlamaCPP engine](https://github.com/ggerganov/llama.cpp) provides an excellent alternative.

Built on the highly optimized llama.cpp library, this engine excels at CPU inference and quantized models, making it particularly suitable for edge deployment and resource-constrained environments. ls even on modest hardware.

LlamaCPP is a great choice for CPU inference and inference with quantized models.

**Installation**

```bash
pip install llama-cpp-python
```

**Basic Usage**

```python
engine = LlamaCppInferenceEngine(
    ModelParams(
        model_name="model.gguf",
        model_kwargs={
            "n_gpu_layers": 0,     # CPU only
            "n_ctx": 2048,         # Context window
            "n_batch": 512,        # Batch size
            "low_vram": True       # Memory optimization
        }
    )
)
```

**Resources**

- [llama.cpp Python Documentation](https://llama-cpp-python.readthedocs.io/en/latest/)
- [llama.cpp GitHub Project](https://github.com/ggerganov/llama.cpp)

### Native Engine

The Native engine uses HuggingFace's [🤗 Transformers](https://huggingface.co/docs/transformers/index) library directly, providing maximum compatibility and ease of use.

While it may not offer the same performance optimizations as vLLM or LlamaCPP, its simplicity and compatibility make it an excellent choice for prototyping and testing.

**Basic Usage**

```python
engine = NativeTextInferenceEngine(
    ModelParams(
        model_name="meta-llama/Llama-3.2-1B-Instruct",
        model_kwargs={
            "device_map": "auto",
            "torch_dtype": "float16"
        }
    )
)
```

**4-bit Quantization**

For memory-constrained environments, 4-bit quantization is available:

```python
model_params = ModelParams(
    model_kwargs={
        "load_in_4bit": True,
    }
)
```

### Remote vLLM

[vLLM](https://github.com/vllm-project/vllm) can be deployed as a server, providing high-performance inference capabilities over HTTP. This section covers different deployment scenarios and configurations.

#### Server Setup

1. **Basic Server** - Suitable for development and testing:

```bash
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --port 6864
```

2. **Multi-GPU Server** - For large models requiring multiple GPUs:

```bash
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.3-70B-Instruct \
    --port 6864 \
    --tensor-parallel-size 4

```

#### Client Configuration

The client can be configured with different reliability and performance options similar to any other remote engine:

```{testcode}
# Basic client with timeout and retry settings
engine = RemoteVLLMInferenceEngine(
    model_params=ModelParams(
        model_name="meta-llama/Llama-3.1-8B-Instruct"
    ),
    remote_params=RemoteParams(
        api_url="http://localhost:6864",
        max_retries=3,      # Maximum number of retries
        num_workers=10,    # Number of parallel threads
    )
)
```

#### Serving LoRA Adapters

Remote vLLM servers can serve LoRA adapters just like local vLLM engines. There are two ways to configure this:

**Option 1: Start Server with LoRA Adapter**

Start the vLLM server with the `--enable-lora` flag and specify the adapter:

```bash
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --port 6864 \
    --enable-lora \
    --lora-modules my-adapter=path/to/lora/adapter
```

Then connect using the adapter name:

```python
engine = RemoteVLLMInferenceEngine(
    model_params=ModelParams(
        model_name="my-adapter"  # Use the adapter name from --lora-modules
    ),
    remote_params=RemoteParams(
        api_url="http://localhost:6864"
    )
)
```

**Option 2: Specify Adapter in Client**

Alternatively, you can specify the `adapter_model` in the client configuration:

```python
engine = RemoteVLLMInferenceEngine(
    model_params=ModelParams(
        model_name="meta-llama/Llama-3.1-8B-Instruct",  # Base model
        adapter_model="path/to/lora/adapter"             # LoRA adapter
    ),
    remote_params=RemoteParams(
        api_url="http://localhost:6864"
    )
)
```

When using `adapter_model` in the client, the adapter path/name will be used as the model identifier in API requests.

**Important Notes:**

- The vLLM server must be started with `--enable-lora` flag to support LoRA adapters
- Multiple LoRA adapters can be served simultaneously from a single server using `--lora-modules`
- Check [vLLM LoRA documentation](https://docs.vllm.ai/en/latest/models/lora.html) for advanced configurations

### Remote SGLang

[SGLang](https://sgl-project.github.io/) is another model server, providing high-performance LLM inference capabilities.

#### Server Setup

```bash
python -m sglang.launch_server \
    --model-path meta-llama/Llama-3.1-8B-Instruct \
    --port 6864 \
    --disable-cuda-graph \
    --mem-fraction-static=0.99
```

Please refer to [SGLang documentation](https://sgl-project.github.io/backend/server_arguments.html) for more advanced configuration options.

#### Client Configuration

The client can be configured with different reliability and performance options similar to any other remote engines:

```{testcode}
engine = SGLangInferenceEngine(
    model_params=ModelParams(
        model_name="meta-llama/Llama-3.1-8B-Instruct"
    ),
    remote_params=RemoteParams(
        api_url="http://localhost:6864"
    )
)
```

To run inference interactively, use the `oumi infer` command with the `-i` flag.

```
oumi infer -c configs/recipes/llama3_1/inference/8b_sglang_infer.yaml -i
```

## Cloud APIs

While local inference offers control and flexibility, cloud APIs provide access to state-of-the-art models and scalable infrastructure without the need to manage your own hardware.

### Anthropic

[Claude](https://www.anthropic.com/claude) is Anthropic's advanced language model, available through their API.

**Basic Usage**

```{testcode}
from oumi.inference import AnthropicInferenceEngine
from oumi.core.configs import ModelParams, RemoteParams

engine = AnthropicInferenceEngine(
    model_params=ModelParams(
        model_name="claude-3-5-sonnet-20240620"
    )
)
```

**Supported Models**

The Anthropic models available via this API as of late Jan'2025 are listed below. For an up-to-date list, please visit [this page](https://docs.anthropic.com/en/docs/about-claude/models).

| Anthropic Model                       | API Model Name            |
|---------------------------------------|---------------------------|
| Claude 3.5 Sonnet (most intelligent)  | claude-3-5-sonnet-latest  |
| Claude 3.5 Haiku (fastest)            | claude-3-5-haiku-latest   |
| Claude 3.0 Opus                       | claude-3-opus-latest      |
| Claude 3.0 Sonnet                     | claude-3-sonnet-20240229  |
| Claude 3.0 Haiku                      | claude-3-haiku-20240307   |

**Resources**

- [Anthropic API Documentation](https://docs.anthropic.com/en/api/getting-started)
- [Available Models](https://docs.anthropic.com/en/docs/about-claude/models)

### Google Cloud

Google Cloud provides multiple pathways for accessing their AI models, either through the Vertex AI platform or directly via the Gemini API.

#### Vertex AI

**Installation**

```bash
pip install "oumi[gcp]"
```

**Basic Usage**

```{testcode}
from oumi.inference import GoogleVertexInferenceEngine
from oumi.core.configs import ModelParams, RemoteParams

engine = GoogleVertexInferenceEngine(
    model_params=ModelParams(
        model_name="google/gemini-1.5-pro"
    ),
    remote_params=RemoteParams(
        api_url="https://{region}-aiplatform.googleapis.com/v1beta1/projects/{project_id}/locations/{region}/endpoints/openapi/chat/completions",
    )
)
```

**Supported Models**

The most popular Google Vertex AI models available via this API (as of late Jan'2025) are listed below. For a full list, including specialized and 3rd party models, please visit [this page](https://cloud.google.com/vertex-ai/generative-ai/docs/learn/models).

| Gemini Model                          | API Model Name                   |
|---------------------------------------|----------------------------------|
| Gemini 2.0 Flash Thinking Mode        | google/gemini-2.0-flash-thinking-exp-01-21 |
| Gemini 2.0 Flash                      | google/gemini-2.0-flash-exp      |
| Gemini 1.5 Flash                      | google/gemini-1.5-flash-002      |
| Gemini 1.5 Pro                        | google/gemini-1.5-pro-002        |
| Gemini 1.0 Pro Vision                 | google/gemini-1.0-pro-vision-001 |

| Gemma Model                           | API Model Name                   |
|---------------------------------------|----------------------------------|
| Gemma 2 2B IT                         | google/gemma2-2b-it              |
| Gemma 2 9B IT                         | google/gemma2-9b-it              |
| Gemma 2 27B IT                        | google/gemma2-27b-it             |
| Code Gemma 2B                         | google/codegemma-2b              |
| Code Gemma 7B                         | google/codegemma-7b              |
| Code Gemma 7B IT                      | google/codegemma-7b-it           |

**Resources**

- [Vertex AI Documentation](https://cloud.google.com/vertex-ai/docs) for Google Cloud AI services

#### Gemini API

**Basic Usage**

```{testcode}
from oumi.inference import GoogleGeminiInferenceEngine
from oumi.core.configs import ModelParams, RemoteParams

engine = GoogleGeminiInferenceEngine(
    model_params=ModelParams(
        model_name="gemini-1.5-flash"
    )
)
```

**Supported Models**

The Gemini models available via this API as of late Jan'2025 are listed below. For an up-to-date list, please visit [this page](https://ai.google.dev/gemini-api/docs/models/gemini).

| Model Name                            | API Model Name            |
|---------------------------------------|---------------------------|
| Gemini 2.0 Flash (experimental)       | gemini-2.0-flash-exp      |
| Gemini 1.5 Flash                      | gemini-1.5-flash          |
| Gemini 1.5 Flash-8B                   | gemini-1.5-flash-8b       |
| Gemini 1.5 Pro                        | gemini-1.5-pro            |
| Gemini 1.0 Pro (deprecated)           | gemini-1.0-pro            |
| AQA                                   | aqa                       |

**Resources**

- [Gemini API Documentation](https://ai.google.dev/docs) for Gemini API details

### OpenAI

[OpenAI's models](https://platform.openai.com/), including GPT-4, represent some of the most widely used and capable AI systems available.

**Basic Usage**

```python
from oumi.inference import OpenAIInferenceEngine
from oumi.core.configs import ModelParams, RemoteParams

engine = OpenAIInferenceEngine(
    model_params=ModelParams(
        model_name="gpt-4o-mini"
    )
)
```

**Supported Models**

The most popular models available via the OpenAI API as of late Jan'2025 are listed below. For a full list please visit [this page](https://platform.openai.com/docs/models)

| OpenAI Model                          | API Model Name            |
|---------------------------------------|---------------------------|
| GPT 4o (flagship model)               | gpt-4o                    |
| GPT 4o mini (fast and affordable)     | gpt-4o-mini               |
| o1 (reasoning model)                  | o1                        |
| o1 mini (reasoning and affordable)    | o1-mini                   |
| GPT-4 Turbo                           | gpt-4-turbo               |
| GPT-4                                 | gpt-4                     |

**Resources**

- [OpenAI API Documentation](https://platform.openai.com/docs) for OpenAI API details

### Together

[Together](https://together.xyz) offers remote inference for 100+ models through serverless endpoints.

**Basic Usage**

```{testcode}
from oumi.inference import TogetherInferenceEngine
from oumi.core.configs import ModelParams, RemoteParams

engine = TogetherInferenceEngine(
    model_params=ModelParams(
        model_name="meta-llama/Llama-3.2-1B-Instruct"
    )
)
```

The models available via this API can be found at [together.ai](https://www.together.ai/).

### Lambda Inference API

[Lambda Inference API](https://lambda.ai) enables you to use large language models (LLMs) without the need to set up a server. No limits are placed on the rate of requests.

**Basic Usage**

```{testcode}
from oumi.inference import LambdaInferenceEngine
from oumi.core.configs import ModelParams, RemoteParams

engine = LambdaInferenceEngine(
    model_params=ModelParams(
        model_name="llama-4-scout-17b-16e-instruct"
    ),
)
```

**Supported Models**

The full list of models available via this API can be found at [docs.lambda.ai](https://docs.lambda.ai/public-cloud/lambda-inference-api/#listing-models).

**Resources**

- [Lambda AI API Documentation](https://docs.lambda.ai/public-cloud/lambda-inference-api)

### DeepSeek

[DeepSeek](https://deepseek.com) allows to access the DeepSeek models (Chat, Code, and Reasoning) through the DeepSeek AI Platform.

**Basic Usage**

```{testcode}
from oumi.inference import DeepSeekInferenceEngine
from oumi.core.configs import ModelParams, RemoteParams

engine = DeepSeekInferenceEngine(
    model_params=ModelParams(
        model_name="deepseek-chat"
    )
)
```

**Supported Models**

The DeepSeek models available via this API as of late Jan'2025 are listed below. For an up-to-date list, please visit [this page](https://api-docs.deepseek.com/quick_start/pricing).

| DeepSeek Model                        | API Model Name            |
|---------------------------------------|---------------------------|
| DeepSeek-V3                           | deepseek-chat             |
| DeepSeek-R1 (reasoning with CoT)      | deepseek-reasoner         |

### SambaNova

[SambaNova](https://www.sambanova.ai/) offers an extreme-speed inference platform on cloud infrastructure with wide variety of models.

This service is particularly useful when you need to run open source models in a managed environment.

**Basic Usage**

```{testcode}
from oumi.inference import SambanovaInferenceEngine
from oumi.core.configs import ModelParams, RemoteParams

engine = SambanovaInferenceEngine(
    model_params=ModelParams(
        model_name="Meta-Llama-3.1-405B-Instruct"
    )
)
```

**Reference**

- [SambaNova's Documentation](https://docs.sambanova.ai/cloud/docs/get-started/overview)

### AWS Bedrock

[AWS Bedrock](https://aws.amazon.com/bedrock/) is Amazon's fully managed service for accessing foundation models from leading AI providers including Anthropic (Claude), Meta (Llama), Amazon (Titan), and more. Bedrock provides a unified API for running inference on these models without managing infrastructure.

**Installation**

```bash
pip install boto3
```

**Setup**

The Bedrock engine requires AWS credentials and the `AWS_REGION` environment variable:

```bash
export AWS_REGION=us-east-1  # or your preferred region
```

Configure AWS credentials using one of these methods:

- AWS CLI: `aws configure`
- Environment variables: `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY`
- IAM roles (for EC2/ECS deployments)

**Basic Usage**

```python
from oumi.inference import BedrockInferenceEngine
from oumi.core.configs import ModelParams, RemoteParams, GenerationParams

engine = BedrockInferenceEngine(
    model_params=ModelParams(
        model_name="anthropic.claude-3-5-sonnet-20240620-v1:0"
    ),
)
```

**Supported Models**

For the complete list of available models and their IDs, visit [AWS Bedrock Model IDs](https://docs.aws.amazon.com/bedrock/latest/userguide/model-ids.html).

**Resources**

- [AWS Bedrock Documentation](https://docs.aws.amazon.com/bedrock/)
- [Bedrock Model IDs](https://docs.aws.amazon.com/bedrock/latest/userguide/model-ids.html)
- [AWS Bedrock Pricing](https://aws.amazon.com/bedrock/pricing/)

### Parasail.io

[Parasail.io](https://parasail.io) offers a cloud-native inference platform that combines the flexibility of self-hosted models with the convenience of cloud infrastructure.

This service is particularly useful when you need to run open source models in a managed environment.

**Basic Usage**

Here's how to configure Oumi for Parasail.io:

```{testcode}
from oumi.inference import ParasailInferenceEngine
from oumi.core.configs import ModelParams, RemoteParams

engine = ParasailInferenceEngine(
    model_params=ModelParams(
        model_name="meta-llama/Llama-3.2-1B-Instruct"
    )
)
```

The models available via this API can be found at [docs.parasail.io](https://docs.parasail.io/).

**Resources**

- [Parasail.io Documentation](https://docs.parasail.io)

## See Also

- [Configuration Guide](configuration.md) for detailed config options
- [Common Workflows](common_workflows.md) for usage examples

# Inference Configuration

## Introduction

This guide covers the configuration options available for inference in Oumi. The configuration system is designed to be:

- **Modular**: Each aspect of inference (model, generation, remote settings) is configured separately
- **Type-safe**: All configuration options are validated at runtime
- **Flexible**: Supports various inference scenarios from local to remote inference
- **Extensible**: Easy to add new configuration options and validate them

The configuration system is built on the {py:obj}`~oumi.core.configs.inference_config.InferenceConfig` class, which contains all inference settings. This class is composed of several parameter classes:

- [Model Configuration](#model-configuration): Model architecture and loading settings via {py:obj}`~oumi.core.configs.params.model_params.ModelParams`
- [Generation Configuration](#generation-configuration): Text generation parameters via {py:obj}`~oumi.core.configs.params.generation_params.GenerationParams`
- [Remote Configuration](#remote-configuration): Remote API settings via {py:obj}`~oumi.core.configs.params.remote_params.RemoteParams`

All configuration files in Oumi are YAML files, which provide a human-readable format for specifying inference settings. The configuration system automatically validates these files and converts them to the appropriate Python objects.

## Basic Structure

A typical configuration file has this structure:

```yaml
model:  # Model settings
  model_name: "meta-llama/Llama-3.1-8B-Instruct"
  trust_remote_code: true
  model_kwargs:
    device_map: "auto"
    torch_dtype: "float16"

generation:  # Generation parameters
  max_new_tokens: 100
  temperature: 0.7
  top_p: 0.9
  batch_size: 1

engine: "VLLM"  # VLLM, LLAMACPP, NATIVE, REMOTE_VLLM, etc.

remote_params:  # Optional remote settings
  api_url: "https://api.example.com/v1"
  api_key: "${API_KEY}"
  connection_timeout: 20.0
```

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

  # Model precision and hardware
  torch_dtype_str: "float32"                         # Model precision (float32/float16/bfloat16/float64)
  device_map: "auto"                                 # Device placement strategy (auto/null)
  compile: false                                     # JIT compile model

  # Attention and optimization
  attn_implementation: null                          # Attention impl (null/sdpa/flash_attention_2/eager)
  enable_liger_kernel: false                         # Enable Liger CUDA kernel for potential speedup

  # Model behavior
  chat_template: null                                # Chat formatting template
  freeze_layers: []                                  # Layer names to freeze during training

  # Additional settings
  model_kwargs: {}                                   # Additional model constructor args
```

#### Using LoRA Adapters

The `adapter_model` parameter allows you to load LoRA (Low-Rank Adaptation) adapters on top of a base model. This is useful when you've fine-tuned a model using LoRA and want to serve the adapted version.

**Configuration Example:**

```yaml
model:
  model_name: "meta-llama/Llama-3.1-8B-Instruct"  # Base model
  adapter_model: "path/to/lora/adapter"           # LoRA adapter path
```

**Engine Support:**

Not all inference engines support LoRA adapters. The following engines support LoRA adapter inference: `VLLM`, `REMOTE_VLLM`, `NATIVE`.

For detailed examples of serving LoRA, see the {doc}`inference_engines` guide.

### Generation Configuration

Configure text generation parameters using the {py:obj}`~oumi.core.configs.params.generation_params.GenerationParams` class:

```yaml
generation:
  max_new_tokens: 256                # Maximum number of new tokens to generate (default: 256)
  batch_size: 1                      # Number of sequences to generate in parallel (default: 1)
  exclude_prompt_from_response: true # Whether to remove the prompt from the response (default: true)
  seed: null                        # Seed for random number determinism (default: null)
  temperature: 0.0                  # Controls randomness in output (0.0 = deterministic) (default: 0.0)
  top_p: 1.0                       # Nucleus sampling probability threshold (default: 1.0)
  frequency_penalty: 0.0           # Penalize repeated tokens (default: 0.0)
  presence_penalty: 0.0            # Penalize tokens based on presence in text (default: 0.0)
  stop_strings: null               # List of sequences to stop generation (default: null)
  stop_token_ids: null            # List of token IDs to stop generation (default: null)
  logit_bias: {}                  # Token-level biases for generation (default: {})
  min_p: 0.0                      # Minimum probability threshold for tokens (default: 0.0)
  use_cache: false                # Whether to use model's internal cache (default: false)
  num_beams: 1                    # Number of beams for beam search (default: 1)
  use_sampling: false             # Whether to use sampling vs greedy decoding (default: false)
  guided_decoding: null           # Parameters for guided decoding (default: null)
  skip_special_tokens: true       # Whether to skip special tokens when decoding (default: true)
```

```{note}
Not all inference engines support all generation parameters. Each engine has its own set of supported parameters which can be checked via the `get_supported_params` attribute of the engine class. For example:
- {py:obj}`NativeTextInferenceEngine <oumi.inference.NativeTextInferenceEngine.get_supported_params>`
- {py:obj}`VLLMInferenceEngine <oumi.inference.VLLMInferenceEngine.get_supported_params>`
- {py:obj}`RemoteInferenceEngine <oumi.inference.RemoteInferenceEngine.get_supported_params>`

Please refer to the specific engine's documentation for details on supported parameters.
```

#### Special Tokens Handling

The `skip_special_tokens` parameter controls whether special tokens (like `<eos>`, `<pad>`, `<bos>`, `<think>`) are included in the decoded output:

- **`true` (default)**: Special tokens are removed from the output, producing clean, readable text suitable for user-facing applications.
- **`false`**: Special tokens are preserved in the output. This is useful for:
  - **Reasoning models**: Models like GPT-OSS (openai/gpt-oss-20b, openai/gpt-oss-120b) that output their internal reasoning using special tokens. Set to `false` to preserve these reasoning tokens.
  - **Tool-calling models**: Models that use special tokens to mark function calls or tool invocations.
  - **Debugging**: When you need to inspect the exact token sequence generated by the model.
  - **Custom parsing**: When implementing custom logic that relies on special tokens in the output format.

```{note}
The `skip_special_tokens` parameter is only supported by {py:obj}`~oumi.inference.NativeTextInferenceEngine` and {py:obj}`~oumi.inference.VLLMInferenceEngine`. Remote API engines typically handle special token filtering automatically and do not expose this parameter.
```

### Remote Configuration

Configure remote API settings using the {py:obj}`~oumi.core.configs.params.remote_params.RemoteParams` class:

```yaml
remote_params:
  api_url: "https://api.example.com/v1"   # Required: URL of the API endpoint
  api_key: "your-api-key"                 # API key for authentication
  api_key_env_varname: null               # Environment variable for API key
  max_retries: 3                          # Maximum number of retries
  connection_timeout: 20.0                # Request timeout in seconds
  num_workers: 1                          # Number of parallel workers
  politeness_policy: 0.0                  # Sleep time between requests
  batch_completion_window: "24h"          # Time window for batch completion
  use_adaptive_concurrency: True          # Whether to change concurrency based on error rate
```

### Engine Selection

The `engine` parameter specifies which inference engine to use. Available options from {py:obj}`~oumi.core.configs.inference_engine_type.InferenceEngineType`:

- `ANTHROPIC`: Use Anthropic's API via {py:obj}`~oumi.inference.AnthropicInferenceEngine`
- `DEEPSEEK`: Use DeepSeek Platform API via {py:obj}`~oumi.inference.DeepSeekInferenceEngine`
- `GOOGLE_GEMINI`: Use Google Gemini via {py:obj}`~oumi.inference.GoogleGeminiInferenceEngine`
- `GOOGLE_VERTEX`: Use Google Vertex AI via {py:obj}`~oumi.inference.GoogleVertexInferenceEngine`
- `LAMBDA`: Use Lambda AI API via {py:obj}`~oumi.inference.LambdaInferenceEngine`
- `LLAMACPP`: Use llama.cpp for CPU inference via {py:obj}`~oumi.inference.LlamaCppInferenceEngine`
- `NATIVE`: Use native PyTorch inference via {py:obj}`~oumi.inference.NativeTextInferenceEngine`
- `OPENAI`: Use OpenAI API via {py:obj}`~oumi.inference.OpenAIInferenceEngine`
- `PARASAIL`: Use Parasail API via {py:obj}`~oumi.inference.ParasailInferenceEngine`
- `REMOTE_VLLM`: Use external vLLM server via {py:obj}`~oumi.inference.RemoteVLLMInferenceEngine`
- `REMOTE`: Use any OpenAI-compatible API via {py:obj}`~oumi.inference.RemoteInferenceEngine`
- `SAMBANOVA`: Use SambaNova API via {py:obj}`~oumi.inference.SambanovaInferenceEngine`
- `SGLANG`: Use SGLang inference engine via {py:obj}`~oumi.inference.SGLangInferenceEngine`
- `TOGETHER`: Use Together API via {py:obj}`~oumi.inference.TogetherInferenceEngine`
- `VLLM`: Use vLLM for optimized local inference via {py:obj}`~oumi.inference.VLLMInferenceEngine`

### Additional Configuration

The following top-level parameters are also available in the configuration:

```yaml
# Input/Output paths
input_path: null    # Path to input file containing prompts (JSONL format)
output_path: null   # Path to save generated outputs
```

The `input_path` should contain prompts in JSONL format, where each line is a JSON representation of an Oumi `Conversation` object.

## See Also

- {doc}`/user_guides/infer/inference_engines` for local and remote inference engines usage
- {doc}`/user_guides/infer/common_workflows` for common workflows
- {doc}`/user_guides/infer/configuration` for detailed parameter documentation

# Inference CLI

## Overview

The Oumi CLI provides a simple interface for running inference tasks. The main command is `oumi infer`,
which supports both interactive chat and batch processing modes. The interactive mode lets you send text inputs
directly from your terminal, while the batch mode lets you submit a jsonl file of conversations for batch processing.

To use the CLI you need an {py:obj}`~oumi.core.configs.InferenceConfig`. This config
will specify which model and inference engine you're using, as well as any relevant
inference-time variables - see {doc}`/user_guides/infer/configuration` for more details.

```{seealso}
Check out our [Infer CLI definition](/cli/commands.md#inference) to see the full list of command line options.
```

## Basic Usage

```bash
# Interactive chat
oumi infer -i -c config.yaml

# Process input file
oumi infer -c config.yaml --input_path input.jsonl --output_path output.jsonl
```

## Command Options

| Option | Description | Default | Example |
|--------|-------------|---------|---------|
| `-c`, `--config` | Configuration file path | Required | `-c config.yaml` |
| `-i`, `--interactive` | Enable interactive mode | False | `-i` |
| `--input_path` | Input JSONL file path | None | `--input_path data.jsonl` |
| `--output_path` | Output JSONL file path | None | `-output_path results.jsonl` |
| `--model.device_map` | GPU device(s) | "cuda" | `--model.device_map "cuda:0"` |
| `--model.model_name` | Model name | None | `--model.model_name "HuggingFaceTB/SmolLM2-135M-Instruct"` |
| `--generation.seed` | Random seed | None | `--seed 42` |
| `--log-level` | Logging level | INFO | `--log-level DEBUG` |

## Configuration File

Example `config.yaml`:

```yaml
model:
  model_name: "meta-llama/Llama-3.1-8B-Instruct"
  model_kwargs:
    device_map: "auto"
    torch_dtype: "float16"

generation:
  max_new_tokens: 100
  temperature: 0.7
  top_p: 0.9
  batch_size: 1

engine: "VLLM"
```

## Common Usage Patterns

### Interactive Chat

```bash
# Basic chat
oumi infer -i -c configs/chat.yaml

# Chat with specific GPU
oumi infer -i -c configs/chat.yaml --model.device_map cuda:0
```

### Batch Processing

```bash
# Process dataset
oumi infer -c configs/batch.yaml \
  --input_path dataset.jsonl \
  --output_path results.jsonl \
  --generation.batch_size 32
```

### Multi-GPU Inference

```bash
# Use specific GPUs
oumi infer -c configs/multi_gpu.yaml \
  --model.device_map "cuda:0,cuda:1"

# Tensor parallel inference
oumi infer -c configs/multi_gpu.yaml \
  --model.model_kwargs.tensor_parallel_size 4
```

## Input/Output Formats

### Input JSONL

```json
{"messages": [{"role": "user", "content": "Hello!"}]}
{"messages": [{"role": "user", "content": "How are you?"}]}
```

### Output JSONL

```json
{"messages": [{"role": "user", "content": "Hello!"}, {"role": "assistant", "content": "Hi!"}]}
{"messages": [{"role": "user", "content": "How are you?"}, {"role": "assistant", "content": "I'm good!"}]}
```

## See Also

- {doc}`configuration` for config file options
- {doc}`common_workflows` for usage examples