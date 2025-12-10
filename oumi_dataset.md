# Dataset Analysis

```{toctree}
:maxdepth: 2
:caption: Dataset Analysis
:hidden:

analyze_config
```

Oumi's dataset analysis framework helps you understand training data before and after fine-tuning. Compute metrics, identify outliers, compare datasets, and create filtered subsets.

**Key capabilities:**

- **Profile datasets**: Understand text length distributions, token counts, and statistics
- **Quality control**: Identify outliers, empty samples, or problematic data
- **Compare datasets**: Analyze multiple datasets with consistent metrics
- **Filter data**: Create filtered subsets based on analysis results

## Quick Start

::::{tab-set-code}
:::{code-block} bash
oumi analyze --config configs/examples/analyze/analyze.yaml
:::
:::{code-block} python
from oumi.core.analyze.dataset_analyzer import DatasetAnalyzer
from oumi.core.configs import AnalyzeConfig, SampleAnalyzerParams

config = AnalyzeConfig(
    dataset_path="data/dataset_examples/oumi_format.jsonl",
    is_multimodal=False,
    analyzers=[SampleAnalyzerParams(id="length")],
)

analyzer = DatasetAnalyzer(config)
analyzer.analyze_dataset()
print(analyzer.analysis_summary)
:::
::::

Oumi outputs results to `./analysis_output/` including per-message metrics, conversation aggregates, and statistical summaries.

## Configuration

A minimal configuration for a local file:

```yaml
dataset_path: data/dataset_examples/oumi_format.jsonl
is_multimodal: false
analyzers:
  - id: length
```

For complete configuration options including dataset sources, output settings, tokenizer configuration, and validation rules, see {doc}`analyze_config`.

## Available Analyzers

### Length Analyzer

The built-in `length` analyzer computes text length metrics:

| Metric | Description |
|--------|-------------|
| `char_count` | Number of characters |
| `word_count` | Number of words (space-separated) |
| `sentence_count` | Number of sentences (split on `.!?`) |
| `token_count` | Number of tokens (requires tokenizer) |

:::{tip}
Enable token counting by adding `tokenizer_config` to your configuration. See {doc}`analyze_config` for setup details.
:::

## Working with Results

### Analysis Summary

Access summary statistics after running analysis:

```python
summary = analyzer.analysis_summary

# Dataset overview
print(f"Dataset: {summary['dataset_overview']['dataset_name']}")
print(f"Samples: {summary['dataset_overview']['conversations_analyzed']}")

# Message-level statistics
for analyzer_name, metrics in summary['message_level_summary'].items():
    for metric_name, stats in metrics.items():
        print(f"{metric_name}: mean={stats['mean']}, std={stats['std']}")
```

### DataFrames

Access raw analysis data as pandas DataFrames:

```python
message_df = analyzer.message_df        # One row per message
conversation_df = analyzer.conversation_df  # One row per conversation
full_df = analyzer.analysis_df          # Merged view
```

### Querying and Filtering

Filter results using pandas query syntax:

```python
# Find long messages
long_messages = analyzer.query("text_content_length_word_count > 10")

# Find short conversations
short_convos = analyzer.query_conversations("text_content_length_char_count < 100")

# Create filtered dataset
filtered_dataset = analyzer.filter("text_content_length_word_count < 100")
```

## Supported Dataset Formats

| Format | Description | Example |
|--------|-------------|---------|
| **oumi** | Multi-turn conversations with roles | SFT, instruction-following |
| **alpaca** | Instruction/input/output format | Stanford Alpaca |
| **DPO** | Preference pairs (chosen/rejected) | Preference learning |
| **KTO** | Binary feedback format | Human feedback |
| **Pretraining** | Raw text | C4, The Pile |

### Analyzing HuggingFace Datasets

Analyze any HuggingFace Hub dataset directly:

::::{tab-set-code}
:::{code-block} yaml

# hf_analyze.yaml

dataset_name: argilla/databricks-dolly-15k-curated-en
split: train
sample_count: 100
output_path: ./analysis_output/dolly
analyzers:

- id: length
:::
:::{code-block} python
from oumi.core.analyze.dataset_analyzer import DatasetAnalyzer
from oumi.core.configs import AnalyzeConfig, SampleAnalyzerParams

config = AnalyzeConfig(
    dataset_name="argilla/databricks-dolly-15k-curated-en",
    split="train",
    sample_count=100,
    analyzers=[SampleAnalyzerParams(id="length")],
)
analyzer = DatasetAnalyzer(config)
analyzer.analyze_dataset()
:::
::::

## Exporting Results

::::{tab-set-code}
:::{code-block} bash

# Export to CSV (default)

oumi analyze --config configs/examples/analyze/analyze.yaml

# Export to Parquet

oumi analyze --config configs/examples/analyze/analyze.yaml --format parquet

# Override output directory

oumi analyze --config configs/examples/analyze/analyze.yaml --output ./my_results
:::
::::

**Output files:**

| File | Description |
|------|-------------|
| `message_analysis.{format}` | Per-message metrics |
| `conversation_analysis.{format}` | Per-conversation aggregated metrics |
| `analysis_summary.json` | Statistical summary |

## Creating Custom Analyzers

You can create custom analyzers to compute domain-specific metrics for your datasets. Custom analyzers extend the `SampleAnalyzer` base class and are registered using the `@register_sample_analyzer` decorator.

For example, to build a question detector analyzer:

```python
import re
from typing import Optional
import pandas as pd

from oumi.core.analyze.column_types import ContentType
from oumi.core.analyze.sample_analyzer import SampleAnalyzer
from oumi.core.registry import register_sample_analyzer


@register_sample_analyzer("questions")
class QuestionAnalyzer(SampleAnalyzer):
    """Counts questions in text fields."""

    def _count_questions(self, text: str) -> int:
        """Count question marks in text. Replace with your own logic."""
        return len(re.findall(r"\?", text))

    def analyze_sample(
        self,
        df: pd.DataFrame,
        schema: Optional[dict] = None,
    ) -> pd.DataFrame:
        result_df = df.copy()

        # Find text columns using the schema
        text_columns = [
            col for col, config in schema.items()
            if config.get("content_type") == ContentType.TEXT and col in df.columns
        ]

        for column in text_columns:
            result_df[f"{column}_question_count"] = (
                df[column].astype(str).apply(self._count_questions)
            )

        return result_df
```

Use your analyzer by referencing its registered ID:

::::{tab-set-code}
:::{code-block} yaml
analyzers:

- id: questions
:::
:::{code-block} python

# Import your analyzer module to trigger registration

import my_analyzers  # noqa: F401

config = AnalyzeConfig(
    dataset_path="data/my_dataset.jsonl",
    is_multimodal=False,
    analyzers=[SampleAnalyzerParams(id="questions")],
)
:::
::::

**Key points:**

- Register with a unique ID via `@register_sample_analyzer("id")`
- Use `schema` to find text columns (`ContentType.TEXT`)
- Prefix output columns with the source column name (e.g., `{column}_question_count`)

## API Reference

- {py:class}`~oumi.core.configs.AnalyzeConfig` - Configuration class
- {py:class}`~oumi.core.analyze.dataset_analyzer.DatasetAnalyzer` - Main analyzer class
- {py:class}`~oumi.core.analyze.sample_analyzer.SampleAnalyzer` - Base class for analyzers
- {py:class}`~oumi.core.analyze.length_analyzer.LengthAnalyzer` - Built-in length analyzer
# Analysis Configuration

{py:class}`~oumi.core.configs.AnalyzeConfig` controls how Oumi analyzes datasets. See {doc}`analyze` for usage examples.

## Core Settings

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `dataset_name` | `str` | Conditional | `None` | Dataset name (HuggingFace Hub or registered) |
| `dataset_path` | `str` | Conditional | `None` | Path to local dataset file |
| `split` | `str` | No | `"train"` | Dataset split to analyze |
| `subset` | `str` | No | `None` | Dataset subset/config name |
| `sample_count` | `int` | No | `None` | Max samples to analyze (None = all) |

## Dataset Specification

Provide either a named dataset or local file path:

::::{tab-set}
:::{tab-item} Named Dataset

```yaml
dataset_name: "argilla/databricks-dolly-15k-curated-en"
split: train
subset: null  # Optional
```

:::
:::{tab-item} Local File

```yaml
dataset_path: data/dataset_examples/oumi_format.jsonl
is_multimodal: false  # Required
```

:::
::::

:::{tip}
You can also pass a pre-loaded dataset directly to `DatasetAnalyzer`:

```python
from oumi.core.analyze.dataset_analyzer import DatasetAnalyzer
analyzer = DatasetAnalyzer(config, dataset=my_dataset)
```

:::

## Output Settings

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `output_path` | `str` | `"."` | Directory for output files |

::::{tab-set-code}
:::{code-block} yaml
output_path: "./analysis_results"
:::
:::{code-block} bash
oumi analyze --config config.yaml --output /custom/path
:::
::::

## Analyzers

Configure analyzers as a list with `id` and optional `params`:

```yaml
analyzers:
  - id: length
    params:
      char_count: true
      word_count: true
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `id` | `str` | Yes | Analyzer identifier (must be registered) |
| `params` | `dict` | No | Analyzer-specific parameters |

### `length` Analyzer

Computes text length metrics:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `char_count` | `bool` | `true` | Character count |
| `word_count` | `bool` | `true` | Word count |
| `sentence_count` | `bool` | `true` | Sentence count |
| `token_count` | `bool` | `false` | Token count (requires tokenizer) |
| `include_special_tokens` | `bool` | `true` | Include special tokens in count |

## Tokenizer Configuration

Required when `token_count: true`:

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `model_name` | `str` | Yes | HuggingFace model/tokenizer name |
| `tokenizer_kwargs` | `dict` | No | Additional tokenizer arguments |
| `trust_remote_code` | `bool` | No | Allow remote code execution |

```yaml
tokenizer_config:
  model_name: openai-community/gpt2
  tokenizer_kwargs:
    use_fast: true
```

## Multimodal Settings

For vision-language datasets:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `is_multimodal` | `bool` | `None` | Whether dataset is multimodal |
| `processor_name` | `str` | `None` | Processor name for VL datasets |
| `processor_kwargs` | `dict` | `{}` | Processor arguments |
| `trust_remote_code` | `bool` | `false` | Allow remote code |

```yaml
dataset_path: "/path/to/vl_data.jsonl"
is_multimodal: true
processor_name: "llava-hf/llava-1.5-7b-hf"
```

:::{note}
Multimodal datasets require a valid `processor_name`.
:::

## Example Configuration

Run the example from the Oumi repository root:

```bash
oumi analyze --config configs/examples/analyze/analyze.yaml
```

The example config at `configs/examples/analyze/analyze.yaml` demonstrates all available options with detailed comments explaining each setting.

## See Also

- {doc}`analyze` - Main analysis guide
- {py:class}`~oumi.core.configs.AnalyzeConfig` - API reference
- {py:class}`~oumi.core.configs.params.base_params.SampleAnalyzerParams` - Analyzer params

# Data Synthesis

The `oumi synth` command enables you to generate synthetic datasets using large language models. Instead of manually creating training data, you can define rules and templates that automatically generate diverse, high-quality examples.

## What You Can Build

- **Question-Answer datasets** for training chatbots
- **Instruction-following datasets** with varied complexity levels
- **Domain-specific training data** (legal, medical, technical)
- **Conversation datasets** with different personas or styles
- **Data augmentation** to expand existing small datasets

## How It Works

The synthesis process follows three steps:

1. **Define attributes** - What varies in your data (topic, difficulty, style, etc.)
2. **Create templates** - How the AI should generate content using those attributes
3. **Generate samples** - The system creates many examples by combining different attribute values

## Your First Synthesis

Let's create a simple question-answer dataset. Save this as `my_first_synth.yaml`:

```yaml
# Generate 10 geography questions
strategy: GENERAL
num_samples: 10
output_path: geography_qa.jsonl

strategy_params:
  # Give the AI an example to learn from
  input_examples:
    - examples:
      - example_question: "What is the capital of France?"

  # Define what should vary across examples
  sampled_attributes:
    - id: difficulty
      name: Difficulty Level
      description: How challenging the question should be
      possible_values:
        - id: easy
          name: Easy
          description: Basic facts everyone should know
        - id: hard
          name: Hard
          description: Detailed knowledge for experts

  # Tell the AI how to generate questions and answers
  generated_attributes:
    - id: question
      instruction_messages:
        - role: SYSTEM
          content: "You are a geography teacher creating quiz questions. Example: {example_question}"
        - role: USER
          content: "Create a {difficulty} geography question. Write the question only, not the answer."
    - id: answer
      instruction_messages:
        - role: SYSTEM
          content: "You are a helpful AI assistant."
        - role: USER
          content: "{question}"

# Configure which AI model to use
inference_config:
  model:
    model_name: claude-3-5-sonnet-20240620
  engine: ANTHROPIC
```

Run it with:

```bash
oumi synth -c my_first_synth.yaml
```

**What happens:** The system will create 10 geography questions, some easy and some hard, saved to `geography_qa.jsonl`.

## Understanding the Results

After running synthesis, you'll see:

- A preview table showing the first few generated samples
- The total number of samples created
- Instructions for using the dataset in training

Each line in the output file contains one example:

```json
{"difficulty": "easy", "question": "What is the largest continent?", "answer": "Asia"}
{"difficulty": "hard", "question": "Which country has the most time zones?", "answer": "France"}
```

## Next Steps: Building More Complex Datasets

Once you're comfortable with the basics, you can create more sophisticated datasets:

### Adding Multiple Attributes

Mix and match different properties (topic + difficulty + style):

```yaml
sampled_attributes:
  - id: topic
    possible_values: [{id: geography}, {id: history}, {id: science}]
  - id: difficulty
    possible_values: [{id: easy}, {id: medium}, {id: hard}]
  - id: style
    possible_values: [{id: formal}, {id: casual}, {id: academic}]
```

### Using Your Own Data

Feed in existing datasets or documents:

```yaml
input_data:
  - path: "my_existing_data.jsonl"
input_documents:
  - path: "textbook.pdf"
```

### Creating Conversations

Build multi-turn dialogues:

```yaml
transformed_attributes:
  - id: conversation
    transformation_strategy:
      type: CHAT
      chat_transform:
        messages:
          - role: USER
            content: "{question}"
          - role: ASSISTANT
            content: "{answer}"
```

Ready to dive deeper? The sections below cover all available options in detail.

---

## Complete Configuration Reference

### Top-Level Parameters

- **`strategy`**: The synthesis strategy to use (currently only `GENERAL` is supported)
- **`num_samples`**: Number of synthetic samples to generate
- **`output_path`**: Path where the generated dataset will be saved (must end with `.jsonl`)
- **`strategy_params`**: Parameters specific to the synthesis strategy
- **`inference_config`**: Configuration for the model used in generation

### Strategy Parameters

The `strategy_params` section defines the core synthesis logic:

#### Input Sources

You can provide data from multiple sources:

**`input_data`**: Existing datasets to sample from

```yaml
input_data:
  - path: "hf:dataset_name"  # HuggingFace dataset
    hf_split: train
  - path: "/path/to/local/data.jsonl"  # Local file
    attribute_map:
      old_column_name: new_attribute_name
```

**`input_documents`**: Documents to segment and use in synthesis

```yaml
input_documents:
  - path: "/path/to/document.pdf"
    id: my_doc
    segmentation_params:
      id: doc_segment
      segment_length: 2048
      segment_overlap: 200
```

**`input_examples`**: Inline examples for few-shot learning

```yaml
input_examples:
  - examples:
    - attribute1: "value1"
      attribute2: "value2"
    - attribute1: "value3"
      attribute2: "value4"
```

#### Attribute Types

**Sampled Attributes**: Randomly selected values from predefined options

```yaml
sampled_attributes:
  - id: difficulty
    name: Difficulty Level
    description: How challenging the question should be
    possible_values:
      - id: easy
        name: Easy
        description: Simple, straightforward questions
        sample_rate: 0.4  # 40% of samples
      - id: medium
        name: Medium
        description: Moderately challenging questions
        sample_rate: 0.4  # 40% of samples
      - id: hard
        name: Hard
        description: Complex, advanced questions
        # No sample_rate specified = 20% (remaining)
```

**Generated Attributes**: Created by LLM using instruction messages

```yaml
generated_attributes:
  - id: summary
    instruction_messages:
      - role: SYSTEM
        content: "You are a helpful summarization assistant."
      - role: USER
        content: "Summarize this text: {input_text}. Format your result as 'Summary: <summary>'"
    postprocessing_params:
      id: clean_summary
      cut_prefix: "Summary: "
      strip_whitespace: true
```

**Transformed Attributes**: Rule-based transformations of existing attributes

```yaml
transformed_attributes:
  - id: conversation
    transformation_strategy:
      type: CHAT
      chat_transform:
        messages:
          - role: USER
            content: "{question}"
          - role: ASSISTANT
            content: "{answer}"
```

#### Advanced Features

**Combination Sampling**: Control probability of specific attribute combinations

```yaml
combination_sampling:
  - combination:
      difficulty: hard
      topic: science
    sample_rate: 0.1  # 10% of samples will have hard science questions
```

**Passthrough Attributes**: Specify which attributes to include in final output

```yaml
passthrough_attributes:
  - question
  - answer
  - difficulty
  - topic
```

## Attribute Referencing

In instruction messages and transformations, you can reference attributes using `{attribute_id}` syntax:

- `{attribute_id}`: The value/name of the attribute
- `{attribute_id.description}`: The description of a sampled attribute value
- `{attribute_id.parent}`: The parent name of a sampled attribute
- `{attribute_id.parent.description}`: The parent description of a sampled attribute

## Postprocessing

Generated attributes can be postprocessed to clean up the output:

```yaml
postprocessing_params:
  id: cleaned_attribute
  keep_original_text_attribute: true  # Keep original alongside cleaned version
  cut_prefix: "Answer: "  # Remove this prefix and everything before it
  cut_suffix: "\n\n"      # Remove this suffix and everything after it
  regex: "\\*\\*(.+?)\\*\\*"  # Extract content between ** **
  strip_whitespace: true  # Remove leading/trailing whitespace
  added_prefix: "Response: "  # Add this prefix
  added_suffix: "."       # Add this suffix
```

## Transformation Strategies

For the following examples, let's assume we have a data sample with the following values.

```json
{
  "question": "What color is the sky?",
  "answer": "The sky is blue."
}
```

### String Transformation

```yaml
transformed_attributes:
  - id: example_string_attribute
    transformation_strategy:
      type: STRING
      string_transform: "Question: {question}\nAnswer: {answer}"
```

Example Result:

```
{
  "example_string_attribute": "Question: What color is the sky?\nAnswer: The sky is blue."
}
```

### List Transformation

```yaml
transformed_attributes:
  - id: example_list_attribute
    transformation_strategy:
      type: LIST
      list_transform:
        - "{question}"
        - "{answer}"
```

Example Result:

```json
{
  "example_list_attribute": [
    "What color is the sky?",
    "The sky is blue.",
  ]
}
```

### Dictionary Transformation

```yaml
transformed_attributes:
  - id: example_dict_attribute
    transformation_strategy:
      type: DICT
      dict_transform:
        question: "{question}"
        answer: "{answer}"
```

Example Result:

```
{
  "example_list_attribute": {
    "question": "What color is the sky?",
    "answer": "The sky is blue.",
  }
}
```

### Chat Transformation

```yaml
transformed_attributes:
  - id: string_attribute
    transformation_strategy:
      type: CHAT
      chat_transform:
        messages:
          - role: USER
            content: "{question}"
          - role: ASSISTANT
            content: "{answer}"
```

## Document Segmentation

When using documents, you can segment them for processing:

```yaml
input_documents:
  - path: "/path/to/document.pdf"
    id: research_paper
    segmentation_params:
      id: paper_segment
      segmentation_strategy: TOKENS
      tokenizer: "openai-community/gpt2"
      segment_length: 1024
      segment_overlap: 128
      keep_original_text: true
```

## Inference Configuration

Configure the model and generation parameters:

```yaml
inference_config:
  model:
    model_name: "claude-3-5-sonnet-20240620"
  engine: ANTHROPIC
  generation:
    max_new_tokens: 1024
    temperature: 0.7
    top_p: 0.9
  remote_params:
    num_workers: 5
    politeness_policy: 60  # Delay between requests in seconds
```

### Supported Engines

- `ANTHROPIC`: Claude models (requires API key)
- `OPENAI`: OpenAI models (requires API key)
- `VLLM`: Local vLLM inference server
- `NATIVE_TEXT`: Local HuggingFace transformers
- And many more (see {doc}`/user_guides/infer/inference_engines`)

## Command Line Options

The `oumi synth` command supports these options:

- `--config`, `-c`: Path to synthesis configuration file (required)
- `--level`: Set logging level (DEBUG, INFO, WARNING, ERROR)

You can also use CLI overrides to modify configuration parameters:

```bash
oumi synth -c config.yaml \
  --num_samples 50 \
  --inference_config.generation.temperature 0.5 \
  --strategy_params.sampled_attributes[0].possible_values[0].sample_rate 0.8
```

## Output Format

The synthesized dataset is saved as a JSONL file where each line contains a JSON object with the attributes in the config:

```json
{"difficulty": "easy", "topic": "geography", "question": "What is the capital of France?", "answer": "Paris"}
{"difficulty": "medium", "topic": "history", "question": "When did World War II end?", "answer": "World War II ended in 1945"}
```

After synthesis completes, you'll see a preview table and instructions on how to use the generated dataset for training:

```
Successfully synthesized 100 samples and saved to synthetic_qa_dataset.jsonl

To train a model, run: oumi train -c path/to/your/train/config.yaml

If you included a 'conversation' chat attribute in your config, update the
config to use your new dataset:
data:
  train:
    datasets:
      - dataset_name: "text_sft_jsonl"
        dataset_path: "synthetic_qa_dataset.jsonl"
```

## Best Practices

1. **Start Small**: Begin with a small `num_samples` to test your configuration
2. **Use Examples**: Provide good examples in `input_examples` for better generation quality
3. **Postprocess Outputs**: Use postprocessing to clean and format generated text
4. **Monitor Costs**: Be aware of API costs when using commercial models
5. **Validate Results**: Review generated samples before using for training
6. **Version Control**: Keep your synthesis configs in version control

## Common Use Cases

### Question-Answer Generation

Generate QA pairs from documents or contexts for training conversational models.

**Example**: See {gh}`configs/examples/synthesis/question_answer_synth.yaml` for a complete geography Q&A generation example.

### Data Augmentation

Create variations of existing datasets by sampling different attributes and regenerating content.

**Example**: See {gh}`configs/examples/synthesis/data_augmentation_synth.yaml` for an example that augments existing datasets with different styles and complexity levels.

### Instruction Following

Generate instruction-response pairs with varying complexity and domains.

**Example**: See {gh}`configs/examples/synthesis/instruction_following_synth.yaml` for a multi-domain instruction generation example covering writing, coding, analysis, and more.

### Conversation Synthesis

Create multi-turn conversations by chaining generated responses.

**Example**: See {gh}`configs/examples/synthesis/conversation_synth.yaml` for a customer support conversation generation example.

### Domain Adaptation

Generate domain-specific training data by conditioning on domain attributes.

**Example**: See {gh}`configs/examples/synthesis/domain_qa_synth.yaml` for a medical domain Q&A generation example with specialty-specific content.

## Troubleshooting

**Empty results**: Check that your instruction messages are well-formed and you have proper API access.

**Slow generation**: Increase `num_workers` or lower `politeness_policy` to improve throughput.

**Out of memory**: Use a smaller model or reduce `max_new_tokens` in generation config.

**Validation errors**: Ensure all attribute IDs are unique and required fields are not empty.

For more help, see the [FAQ](../faq/troubleshooting.md) or report issues at https://github.com/oumi-ai/oumi/issues.