# Evaluation Configuration

Oumi allows users to define their evaluation configurations through a `YAML` file, providing a flexible, human-readable, and easily customizable format for setting up experiments. By using `YAML`, users can effortlessly configure model and generation parameters, and define a list of tasks to evaluate with. This approach not only streamlines the process of configuring evaluations but also ensures that configurations are easily versioned, shared, and reproduced across different environments and teams.

# Configuration Structure

The configuration `YAML` file is loaded into {py:class}`~oumi.core.configs.EvaluationConfig` class, and consists of {py:class}`~oumi.core.configs.params.model_params.ModelParams`, {py:class}`~oumi.core.configs.params.evaluation_params.EvaluationTaskParams`, and {py:class}`~oumi.core.configs.params.generation_params.GenerationParams`. If the evaluation benchmark is generative, meaning that the model responses need to be first generated (inferred) and then evaluated by a judge, you can also set the `inference_engine` ({py:obj}`~oumi.core.configs.inference_engine_type.InferenceEngineType`) for local inference or the `inference_remote_params` ({py:obj}`~oumi.core.configs.params.remote_params.RemoteParams`) for remote inference.

Here's an advanced configuration example, showing many of the available parameters:

```yaml
model:
  model_name: "microsoft/Phi-3-mini-4k-instruct"
  trust_remote_code: True
  adapter_model: "path/to/adapter"  # Optional: For adapter-based models

tasks:
  # LM Harness Tasks
  - evaluation_backend: lm_harness
    task_name: mmlu
    num_samples: 100
    eval_kwargs:
      num_fewshot: 5
  - evaluation_backend: lm_harness
    task_name: arc_challenge
    eval_kwargs:
      num_fewshot: 25
  - evaluation_backend: lm_harness
    task_name: hellaswag
    eval_kwargs:
      num_fewshot: 10

  # AlpacaEval Task
  - evaluation_backend: alpaca_eval
    version: 2.0  # or 1.0
    num_samples: 805

  # Custom Task
  - evaluation_backend: custom
    task_name: my_custom_evaluation

generation:
  batch_size: 16
  max_new_tokens: 512
  temperature: 0.0

inference_engine: NATIVE

output_dir: "my_evaluation_results"
enable_wandb: true
run_name: "phi3-evaluation"
```

# Configuration Options

- `model`: Model-specific configuration ({py:class}`~oumi.core.configs.params.model_params.ModelParams`)
  - `model_name`: HuggingFace model identifier or local path
  - `trust_remote_code`: Whether to trust remote code (for custom models)
  - `adapter_model`: Path to adapter weights (optional)
  - `adapter_type`: Type of adapter ("lora" or "qlora")
  - `shard_for_eval`: Enable multi-GPU parallelization on a single node

- `tasks`: List of evaluation tasks ({py:class}`~oumi.core.configs.params.evaluation_params.EvaluationTaskParams`)
  - LM Harness Task Parameters:   ({py:class}`~oumi.core.configs.params.evaluation_params.LMHarnessTaskParams`)
    - `evaluation_backend`: "lm_harness"
    - `task_name`: Name of the LM Harness task
    - `num_fewshot`: Number of few-shot examples (0 for zero-shot)
    - `num_samples`: Number of samples to evaluate
    - `eval_kwargs`: Additional task-specific parameters

  - AlpacaEval Task Parameters: ({py:class}`~oumi.core.configs.params.evaluation_params.AlpacaEvalTaskParams`)
    - `evaluation_backend`: "alpaca_eval"
    - `version`: AlpacaEval version (1.0 or 2.0)
    - `num_samples`: Number of samples to evaluate
    - `eval_kwargs`: Additional task-specific parameters

  - Custom Task Parameters:
    - `evaluation_backend`: "custom"
    - `task_name`: Name that the custom evaluation function was registered with

- `generation`: Generation parameters ({py:class}`~oumi.core.configs.params.generation_params.GenerationParams`)
  - `batch_size`: Batch size for inference ("auto" for automatic selection)
  - `max_new_tokens`: Maximum number of tokens to generate
  - `temperature`: Sampling temperature

- `inference_engine`: Inference engine for local inference ({py:obj}`~oumi.core.configs.inference_engine_type.InferenceEngineType`)
- `inference_remote_params`: Inference parameters for remote inference ({py:obj}`~oumi.core.configs.params.remote_params.RemoteParams`)

- `enable_wandb`: Enable Weights & Biases logging
- `output_dir`: Directory for saving results
- `run_name`: Name of the evaluation run

# Standardized Benchmarks

Standardized benchmarks are important for evaluating LLMs because they provide a consistent and objective way to compare the performance of different models across various tasks. This allows researchers and developers to accurately assess progress, identify strengths and weaknesses, all while ensuring fair comparisons between different LLMs.

## Overview

These benchmarks assess a model's general and domain-specific knowledge, its comprehension and ability for commonsense reasoning and logical analysis, entity recognition, factuality and truthfulness, as well as mathematical and coding capabilities. In standardized benchmarks, the prompts are structured in a way so that possible answers can be predefined.

The most common method to limit the answer space for standardized tasks is asking the model to select the correct answer from set of multiple-choice options (e.g., A, B, C, D), based on its understanding and reasoning about the input. Another way is limiting the answer space to a single word or a short phrase, which can be directly extracted from the text. In this case, the model's task is to identify the correct word/phrase that answers a question or matches the entity required. An alternative setup is asking the model to chronologically rank a set of statements, rank them to achieve logical consistency, or rank them on metrics such as plausibility/correctness, importance, or relevance. Finally, fill-in-the-blank questions, masking answer tasks, and True/False questions are also popular options for limiting the answer space.

Oumi uses EleutherAI’s [LM Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness) to power scalable, high-performance evaluations of LLMs, providing robust and consistent benchmarking across a wide range of [standardized tasks](https://github.com/EleutherAI/lm-evaluation-harness/tree/main/lm_eval/tasks).

## Popular Benchmarks

This section discusses the most popular standardized benchmarks, in order to give you a starting point for your evaluations. You can kick-off evaluations using the following configuration template. For advanced configuration settings, please visit the {doc}`evaluation configuration </user_guides/evaluate/evaluation_config>` page.

```yaml
model:
  model_name: <HuggingFace model name or local path to model>
  trust_remote_code: False # Set to true for HuggingFace models

tasks:
  - evaluation_backend: lm_harness
    task_name: <`Task Name` from the tables below>
    eval_kwargs:
      num_fewshot: <number of few-shot prompts, if applicable>

output_dir: <output directory>
```

To see all supported standardized benchmarks:

```bash
lm-eval --tasks list
```

### Question Answering and Knowledge Retrieval
Benchmarks that evaluate a model's ability to understand questions and generate accurate answers, based on the provided context (Open-Book) or its internal knowledge (Closed-Book).

| Task | Description | Type | Task Name | Introduced |
|------|-------------|------|-----------|------------|
BoolQ (Boolean Questions) | A question-answering task consisting of a short passage from a Wikipedia article and a yes/no question about the passage [[details](https://arxiv.org/abs/1905.00537)] | Open-Book (True/False answer) | `boolq` | 2019, as part of Superglue
TriviaQA | Trivia question answering to test general knowledge, using evidence documents [[details](https://nlp.cs.washington.edu/triviaqa/)] | Open-Book (free-form answer) | `triviaqa` | 2017, by UW in ACL
CoQA (Conversational Question Answering) | Measure the ability of machines to understand a text passage and answer a series of interconnected questions that appear in a conversation [[details](https://arxiv.org/abs/1808.07042)] | Open-Book (free-form answer) | `coqa` | 2018, by Stanford in TACL
NQ (Natural Questions) | Open domain question answering benchmark that is derived from Natural Questions. The goal is to predict an answer for a  question in English [[details](https://research.google/pubs/natural-questions-a-benchmark-for-question-answering-research/)] | Closed-Book (free-form answer) | `nq_open` | 2019, by Google in TACL
SQuAD V2 (Stanford Question Answering Dataset) | Reading comprehension dataset, consisting of questions posed by crowdworkers on a set of Wikipedia articles. The answer is either a segment of text from the reading passage or unanswerable [[details](https://arxiv.org/abs/1806.03822)] | Open-Book (free-form answer) | `squadv2` | 2018, by Stanford in ACL
GPQA (Google-Proof Q&A) | Very difficult multiple-choice questions written by domain experts in biology, physics, and chemistry [[details](https://arxiv.org/abs/2311.12022)] | Closed-Book (multichoice answer) | `gpqa` | 2023, by NYU, Cohere, Anthropic
ARC Challenge (AI2 Reasoning Challenge) | Challenging multiple-choice science questions from the ARC dataset. Answered incorrectly by standard retrieval-based and word co-occurrence algorithms [[details](https://arxiv.org/abs/1803.05457)] | Closed-Book (multichoice answer) | `arc_challenge` | 2018, by Allen AI
MMLU (Massive Multitask Language Understanding) | Multiple choice QA benchmark on elementary mathematics, US history, computer science, law, and more [[details](https://arxiv.org/abs/2009.03300)] | Closed-Book (multichoice answer) | `mmlu` | 2021, by Berkeley, Columbia and others in ICLR
MMLU Pro (Massive Multitask Language Understanding) | Enhanced MMLU extending the knowledge-driven questions by integrating more challenging, reasoning-focused questions and expanding the choice set from four to ten options [[details](https://arxiv.org/abs/2406.01574)] | Closed-Book (multichoice answer) | `mmlu_pro` | 2024, by U Waterloo, U Toronto, CMU
Truthful QA | Measures if model mimics human falsehoods. Assesses truthfulness and ability to avoid humans' false beliefs or misconceptions (38 categories, including health, law, finance and politics) [[details](https://arxiv.org/abs/2109.07958)] | Open-Book (both multichoice and free-form) | `truthfulqa_mc2` | 2022, by University of Oxford, OpenAI

### Commonsense and Logical Reasoning
Benchmarks that assess a model's ability to perform reasoning tasks requiring commonsense understanding and logical thinking.

| Task | Description | Task Name | Introduced |
|------|-------------|-----------|------------|
Commonsense QA | Multiple-choice question answering dataset that requires different types of commonsense knowledge to predict the correct answer among one correct answer and four distracting answers [[details](https://arxiv.org/pdf/1811.00937.pdf)] | `commonsense_qa` | 2019, by Tel-Aviv University and Allen AI
PIQA (Physical Interaction QA) | Physical commonsense reasoning to investigate the physical knowledge of existing models, including basic properties of the real-world objects  [[details](https://arxiv.org/abs/1911.11641)] | `piqa` | 2019, by Allen AI, MSR, CMU, UW
SocialIQA (Social Interaction QA) | Commonsense reasoning about social situations, probing emotional and social intelligence in a variety of everyday situations [[details](https://arxiv.org/abs/1904.09728)] | `siqa` | 2019, by Allen AI, UW
SWAG (Situations With Adversarial Generations) | Grounded commonsense reasoning. Questions sourced from video captions with answers being what might happen next in the next scene (1 correct and 3 adversarially generated choices) [[details](https://arxiv.org/abs/1808.05326)] | `swag` | 2019, by UW, Allen AI
HellaSWAG | Benchmark that builds on SWAG to evaluate understanding and common sense reasoning, particularly in the context of completing sentences or narratives [[details](https://arxiv.org/abs/1905.07830)] | `hellaswag` | 2019, by UW, Allen AI
WinoGrande | Given a sentence which requires commonsense reasoning, choose the right option among multiple choices. Inspired by Winograd Schema Challenge (WSC) [[details](https://arxiv.org/abs/1907.10641)] | `winogrande` | 2019, by Allen AI
MuSR (Multistep Soft Reasoning) | Multistep soft reasoning tasks specified in a natural language narrative. Includes solving murder mysteries, object placement, and team allocation [[details](https://arxiv.org/abs/2310.16049)] | `leaderboard_musr` | 2024, by UT Austin in ICLR
DROP (Discrete Reasoning Over Paragraphs) | Reading comprehension benchmark. Requires reference resolution and performing discrete operations over the references (addition, counting, or sorting) [[details](https://arxiv.org/abs/1903.00161)] | `drop` | 2019, by UC Irvine, Peking University and others
ANLI (Adversarial NLI) | Reasoning dataset. Given a premise, identify if a hypothesis is entailment, neutral, or contradictory [[details](https://arxiv.org/abs/1910.14599)] | `anli` | 2019, by UNC Chapel Hill and Meta
BBH (Big Bench Hard) | Challenging tasks from the BIG-Bench evaluation suite, focusing on complex reasoning, multi-step problem solving, and requiring deep document understanding rather than surface-level pattern matching [[details](https://arxiv.org/abs/2210.09261)] | `bbh` | 2022, by Google Research and Stanford

### Language Understanding
Benchmarks that test a model's understanding of language semantics and syntax.

| Task | Description | Task Name | Introduced |
|------|-------------|-----------|------------|
WiC (Words in Context) | Word sense disambiguation task. Requires identifying if occurrences of a word in two contexts correspond to the same meaning or not. Framed as a binary classification task [[details](https://arxiv.org/abs/1905.00537)] | `wic` | 2019, as part of Superglue
RTE (Recognizing Textual Entailment) | Given two text fragments, recognize whether the meaning of one fragment can be inferred from the other [[details](https://arxiv.org/abs/1905.00537)] | `rte` | 2019, as part of Superglue
LAMBADA (LAnguage Modeling Broadened to Account for Discourse Aspects) | Word prediction task. Given a passage, predict the last word. Requires tracking information in the broader discourse, beyond the last sentence [[details](https://arxiv.org/abs/1606.06031)] | `lambada` | 2016, by CIMeC, University of Trento
WMT 2016 (Workshop on Machine Translation) | Collection of parallel text data used to assess the performance of machine translation systems, primarily focusing on news articles, across various language pairs [[details](http://www.aclweb.org/anthology/W/W16/W16-2301)] | `wmt16` | 2016, by Charles University, FBK, and others
RACE (ReAding Comprehension from Examinations) | Reading comprehension dataset collected from English examinations in China. Designed for middle school and high school students. Evaluates language understanding and reasoning [[details](https://arxiv.org/abs/1704.04683)] | `race` | 2017, by CMU
IFEval (Instruction Following Evaluation) | Instruction-Following evaluation dataset. Focuses on formatting text, including imposing length constraints, paragraph composition, punctuation, enforcing lower/upper casing, including/exluding keywords, etc [[details](https://arxiv.org/abs/2311.07911)] | `ifeval` | 2023, by Google, Yale University
<!-- FIXME: Move IFEval to generative Benchmarks-->

### Mathematical and Numerical Reasoning
Benchmarks focused on evaluating a model's ability to perform mathematical calculations and reason about numerical information.

| Task | Description | Task Name | Introduced |
|------|-------------|-----------|------------|
MATH (Mathematics Aptitude Test of Heuristics), Level 5  | Challenging competition mathematics problems that require step-by-step solutions [[details](https://arxiv.org/abs/2103.03874)] | `leaderboard_math_hard` | 2021, by UC Berkeley
GSM 8K (Grade School Math) | Grade school-level math word problems [[details](https://arxiv.org/abs/2110.14168)] | `gsm8k` | 2021, by OpenAi

(multi-modal-standardized-benchmarks)=
### Multi-modal Benchmarks

Benchmarks to evaluate vision-language (image + text) models.

| Task | Description | Task Name | Introduced |
|------|-------------|-----------|------------|
MMMU (Massive Multi-discipline Multimodal Understanding) | Designed to evaluate multimodal (image + text) models on multi-discipline tasks demanding college-level subject knowledge and deliberate reasoning. MMMU includes 11.5K meticulously collected multimodal questions from college exams, quizzes, and textbooks, covering six core disciplines [[details](https://arxiv.org/abs/2311.16502)] | `mmmu_val` | 2023, by OSU and others

## Trade-offs

### Advantages

The closed nature of standardized benchmarks allows for more precise and objective evaluation, focusing on a model's ability to understand, reason, and extract information accurately. The benchmarks assess a wide range of model skills in a controlled and easily quantifiable way.

1. **Objective and consistent evaluation**. With a closed answer space, there are no subjective interpretations of what constitutes the correct answer, since there’s a clear right answer among a set of predefined choices. This ensures consistency in scoring, allowing evaluators to use standard metrics (F1 score, precision, recall, accuracy, etc.) in a straightforward manner. In addition, results from different models can be directly compared because the possible answers are fixed, ensuring consistency across evaluations.

2. **Reproducibility**. When models are tested on the same benchmark with the same set of options, other researchers can replicate the results and verify claims, as long as (i) all the environmental settings are the same (Oumi thoroughly logs all settings that could affect evaluation variability) and (ii) the model is prompted with temperature 0.0 and a consistent seed. Reproducibility is crucial to track improvements across models or versions, as well as scientific rigor and advancing the state of the art in AI research.

3. **Task and domain diversity**. These benchmarks have very wide coverage and include a broad spectrum of tasks, which can highlight specific areas where a model excels or falls short. They reflect real-world challenges and complexities. There is also a multitude of benchmarks that test a model on domain-specific intricacies, assessing its ability to apply specialized knowledge within a particular field, ensuring that evaluation is closely tied to practical performance.

4. **Low cost inference and development**. In closed spaces, the model's output is often a straightforward prediction (e.g., a multiple choice letter or a single word), which is less resource-intensive since it only requires generating a few tokens (vs. a complex full-text response). In addition, the model doesn't need to consider an infinite range of possible responses, it can focus its reasoning or search on a smaller, fixed set of options, also contributing in faster inference. Developing such benchmarks also involves a simpler annotation process and low-cost labelling.

### Limitations

While standardized benchmarks offer several advantages, they also come with several limitations compared to generative benchmarks, especially in assessing the broader, more complex language abilities that are required in many real-world applications such as creativity or nuanced reasoning.

1. **Open-ended problem solving and novelty**: Models are not tested on their ability to generate creative or novel responses, explain the steps required to address a problem, being aware of the previous context to keep a conversation engaging, or to handle tasks where there isn’t a single correct answer. Many real-world applications, such as conversational agents, generating essays and stories, or summarization demand open-ended problem solving.

2. **Language quality and human alignment**. In tasks that require text generation, the style, fluency, and coherence of a model's output are crucial. Closed-answer benchmarks do not assess how well a model can generate meaningful, varied, or contextually rich language. Adapting to a persona or tone, if requested by the user, is also not assessed. Finally, alignment with human morals and social norms, being diplomatic when asked controversial questions, understanding humor and being culturally aware are outside the scope of standardized benchmarks.

3. **Ambiguity**. Closed-answer benchmarks do not evaluate the model's ability to handle ambiguous prompts. This is a common real-word scenario and an important conversational skill for agents. Addressing ambiguity typically involves asking for clarifications, requesting more context, or engaging in a dynamic context-sensitive back-and-forth conversation with targeted questions until the user's intention is revealed and becomes clear and actionable.

4. **Overfitting and cheating**. Boosting performance on standardized benchmarks requires that the model is trained on similar benchmarks. However, since the answer space is fixed and closed, models may overfit and learn to recognize patterns that are only applicable to multiple choice answers, struggling to generalize in real-world scenarios where the "correct" answer isn’t part of a predefined set. In addition, intentionally or unintentionally training on the test set is an emerging issue, which is recently (only partially) addressed by contamination IDs.

<!-- suggesting to DROP this until we fully support it; currently it hurts more than helps IMO:

## Custom LM-Harness Tasks

While Oumi provides integration with the LM Evaluation Harness and its extensive task collection, you may need to create a custom evaluation tasks for specific use cases. For this case, we refer you to the [new task guide](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/new_task_guide.md), which walks you through the process of creating and implementing custom evaluation tasks using the `LM Evaluation Harness` (`lm_eval`) framework.

-->
# Generative Benchmarks

Evaluating language models on open-ended generation tasks requires specialized approaches beyond standardized benchmarks and traditional metrics. In this section, we discuss several established methods for assessing model performance with generative benchmarks. Generative benchmarks consist of open-ended questions, allowing the model to generate a free-form output, rather than adhere to a pre-defined "correct" answer. The focus of such evaluations is to assess the model's ability to follow the prompt instructions and generate human-like, high-quality, coherent, and creative responses.

## Overview

Generative benchmarks are vital to evaluate conversational agents, as well as tasks such as creative writing or editing (storytelling, essays and articles), summarization, translation, planning, and code generation. In addition, assessing capabilities such as instruction following, safety, trust, and groundedness, require generative responses.

That said, generative evaluations are significantly more challenging than closed-form evaluations, due to lack of a clear "correct" answer. This makes the evaluation criteria subjective to human judgment. But, even for an established set of criteria, aligning across raters ultimately depends on human perception, making consistent evaluations a very hard problem. Alternatively, fully-automating the rating process, by leveraging LLMs as judges of responses is recently getting more traction. LLM-as-a-judge platforms are significantly more cost- and time-effective, while they can provide reproducible and consistent results (under certain conditions).

This section discusses the LLM-as-a-judge platforms that Oumi is using as its backend to provide reliable insights on generative model performance. The evaluation process consists of 2 steps: inference and judgement. Inference generates model responses for a predefined set of open-ended prompts, while judgement leverages an LLM to judge the quality of these responses. Oumi enables generative evaluation by integrating with popular platforms (AlpacaEval and MT-Bench), as well as offering a flexible framework (see {doc}`/user_guides/judge/judge`) for users to develop their own generative evaluations.

All evaluations in Oumi are automatically logged and versioned, capturing model configurations, evaluation parameters, and environmental details to ensure reproducible results.

## Supported Out-of-the-box Benchmarks

### AlpacaEval (1.0 and 2.0)

[AlpacaEval](https://github.com/tatsu-lab/alpaca_eval) is a framework for automatically evaluating the instruction-following capabilities of language models, as well as whether their responses are helpful, accurate, and relevant. The framework prioritizes human-aligned evaluation, aiming to assess whether the model’s response meets the expectations of human evaluators. The instruction set consists of 805 open-ended questions, such as "How did US states get their names?".

The latest update (2.0) uses GPT-4 Turbo as a judge, comparing the model outputs against a set of reference responses, and calculating standardized win-rates against these responses. AlpacaEval 2.0 has been widely adopted as a benchmark in research papers and it is particularly useful for evaluating instruction-tuned models, comparing performance against established baselines (see [leaderboard](https://tatsu-lab.github.io/alpaca_eval/)), and conducting automated evaluations at scale.

To use AlpacaEval, you can run the following command:

```bash
OPENAI_API_KEY="your_key"
oumi evaluate -c configs/recipes/smollm/evaluation/135m/quickstart_alpaca_v2_eval.yaml
```

If you prefer to use AlpacaEval outside Oumi, we refer you to our example notebook {gh}`notebooks/Oumi - Evaluation with AlpacaEval 2.0.ipynb`.

**Resources:**
- [AlpacaEval V1.0 Paper](https://arxiv.org/abs/2305.14387)
- [AlpacaEval V2.0 Paper](https://arxiv.org/abs/2404.04475)
- [AlpacaEval V2.0 Dataset](https://huggingface.co/datasets/tatsu-lab/alpaca_eval)
- [Leaderboard](https://tatsu-lab.github.io/alpaca_eval/)
- [Official Repository](https://github.com/tatsu-lab/alpaca_eval)

### MT-Bench

[MT-Bench](https://github.com/lm-sys/FastChat/tree/main/fastchat/llm_judge) (Multi-Turn Benchmark) is an evaluation framework specifically designed for assessing chat assistants in multi-turn conversations. It tests a model's ability to maintain context, provide consistent responses across turns, and engage in coherent dialogues. The instruction set consists of 80 open-ended multi-turn questions, which span across 8 popular categories: writing, roleplay, extraction, reasoning, math, coding, STEM knowledge, knowledge of social sciences.

MT-Bench uses GPT-4 as a judge to score each answer on a scale of 10, or perform pairwise scoring between 2 models, and calculates standardized win-rates. It can also breakdown the scoring per category (see [notebook](https://colab.research.google.com/drive/15O3Y8Rxq37PuMlArE291P4OC6ia37PQK#scrollTo=5i8R0l-XqkgO)). Overall, it offers several key features including multi-turn conversation evaluation with increasing complexity, diverse question categories spanning various domains, and a standardized scoring system powered by GPT-4 judgments.

To evaluate a model with MT-Bench, see the example notebook {gh}`notebooks/Oumi - Evaluation with MT Bench.ipynb`.

**Resources:**
- {gh}`MT-Bench Tutorial <notebooks/Oumi - Evaluation with MT Bench.ipynb>`
- [MT-Bench Paper](https://arxiv.org/abs/2306.05685)
- [Human Annotation Dataset](https://huggingface.co/datasets/lmsys/mt_bench_human_judgments)
- [Leaderboard](https://huggingface.co/spaces/lmarena-ai/chatbot-arena-leaderboard)
- [Official Repository](https://github.com/lm-sys/FastChat/tree/main/fastchat/llm_judge)

<!--- Commented; we do NOT support HumanEval yet.
### HumanEval

HumanEval is a benchmark designed to evaluate language models' capabilities in generating functional code from natural language descriptions. It consists of programming challenges that test both understanding of requirements and ability to generate correct, efficient code solutions.

**Resources:**
- [HumanEval Paper](https://arxiv.org/abs/2107.03374)
- [Official Repository](https://github.com/openai/human-eval)
- [Dataset Documentation](https://huggingface.co/datasets/openai_humaneval)
-->

## LLM-as-a-judge

While the out-of-the-box benchmarks provided by Oumi cover a broad spectrum of generative use cases, we understand that many specialized applications require more tailored evaluation objectives. If the existing benchmarks do not fully meet your needs, Oumi offers a flexible and streamlined process to create and automate evaluations, by leveraging an {doc}`LLM Judge </user_guides/judge/judge>`.

You can author your own set of evaluation prompts and customize the metrics to align with your specific domain or use case. By leveraging an LLM to assess your model's outputs, you can fully automate the evaluation pipeline, producing insightful scores that truly reflect your unique criteria.

**Resources:**
- {gh}`Simple Judge <notebooks/Oumi - Simple Judge.ipynb>` notebook
erboards provide a structured, transparent, and competitive environment for evaluating Large Language Models (LLMs), helping to guide the development of more powerful, reliable, and useful models while fostering collaboration and innovation within the field. This page discusses how to evaluate models on popular leaderboards.

HuggingFace Leaderboard V2
As of early 2025, the most popular standardized benchmarks, used across academia and industry, are the benchmarks introduced by HuggingFace’s latest (V2) leaderboard. HuggingFace has posted a blog elaborating on why these benchmarks have been selected, while EleutherAI has also provided a comprehensive README discussing the benchmark evaluation goals, coverage, and applicability.

MMLU-Pro (Massive Multitask Language Understanding) [paper]

GPQA (Google-Proof Q&A Benchmark) [paper]

MuSR (Multistep Soft Reasoning) [paper]

MATH (Mathematics Aptitude Test of Heuristics, Level 5). [paper]

IFEval (Instruction Following Evaluation) [paper]

BBH (Big Bench Hard) [paper]

You can evaluate a model on Hugging Face’s latest leaderboard by creating a yaml file and invoking the CLI with the following command:

oumi launch up -c configs/recipes/smollm/evaluation/135m/leaderboards/huggingface_leaderboard_v2_eval.yaml
A few things to pay attention to:

GPQA Gating. Access to GPQA is restricted through gating mechanisms, to minimize the risk of data contamination. Before running the leaderboard evaluation, you must first log in to HuggingFace and accept the terms of use for QPQA. In addition, you need to authenticate on the Hub using HuggingFace’s User Access Token when launching the evaluation job. You can do so either by setting the environmental HuggingFace token variable HF_TOKEN or by storing its value at HF_TOKEN_PATH (default location is ~/.cache/huggingface/token).

Dependencies. This leaderboard (specifically the IFEval and MATH benchmarks) requires specific packages to be deployed to function correctly. You can either install all Oumi evaluation packages with pip install oumi[evaluation], or explore the required packages for each benchmark at oumi-ai/oumi and only install the packages needed for your specific case.

HuggingFace Leaderboard V1
Before HuggingFace’s leaderboard V2 was introduced, the most popular benchmarks were captured in the V1 leaderboard. Note that due to the fast advancement of AI models, many of these benchmarks have been saturated (i.e., they became too easy to measure meaningful improvements for recent models) while newer models also showed signs of contamination, indicating that data very similar to these benchmarks may exist in their training sets.

ARC (AI2 Reasoning Challenge) [paper]

MMLU (Massive Multitask Language Understanding) [paper]

Winogrande (Adversarial Winograd Schema Challenge at Scale) [paper]

HellaSwag (Harder Endings, Longer contexts, and Low-shot Activities for Situations With Adversarial Generations) [paper]

GSM 8K (Grade School Math) [paper]

TruthfulQA (Measuring How Models Mimic Human Falsehoods) [paper]

You can evaluate a model on Hugging Face’s V1 leaderboard by creating a yaml file and invoking the CLI with the following command:

oumi launch up -c configs/recipes/smollm/evaluation/135m/leaderboards/huggingface_leaderboard_v1_eval.yaml
Running Remotely
Running leaderboard evaluations can be resource-intensive, particularly when working with large models that require GPU acceleration. As such, you may need to execute on remote machines with the necessary hardware resources. Provisioning and running leaderboard evaluations on a remote GCP machine can be achieved with the following sample yaml code.

HuggingFace Leaderboard V2:

oumi launch up -c configs/recipes/smollm/evaluation/135m/leaderboards/huggingface_leaderboard_v2_gcp_job.yaml
HuggingFace Leaderboard V1:

oumi launch up -c configs/recipes/smollm/evaluation/135m/leaderboards/huggingface_leaderboard_v1_gcp_job.yaml
Tip

In addition to GCP, Oumi supports out-of-the-box various cloud providers (including AWS, Azure, Runpod, Lambda) or even your own custom cluster. To explore these, visit the running code on clusters page.

A few things to pay attention to:

Output folder. When executing in a remote machine that is not accessible after the evaluation completes, you need to re-direct your output to persistent storage. For GCP, you can store your output into a mounted GCS Bucket. For example, assuming your bucket is gs://my-gcs-bucket, mount to it and set output_dir as shown below.

storage_mounts:
  /my-gcs-bucket:
    source: gs://my-gcs-bucket
    store: gcs

output_dir: "/my-gcs-bucket/huggingface_leaderboard"
HuggingFace Access Token. If you need to authenticate on the HuggingFace Hub to access private or gated models, datasets, or other resources that require authorization, you need to cache HuggingFace’s User Access Token in the remote machine. This token is acting as a HuggingFace login credential to interact with the platform beyond publicly available content. To do so, mount the locally cached token file (by default ~/.cache/huggingface/token) to the remote machine, as shown below.

file_mounts:
  ~/.cache/huggingface/token: ~/.cache/huggingface/token # HF credentials
W&B Credentials. If you are using Weights & Biases for experiment tracking, make sure you mount the locally cached credentials file (by default ~/.netrc) to the remote machine, as shown below.

file_mounts:
  ~/.netrc: ~/.netrc
Dependencies. If you need to deploy packages in the remote machine, such as Oumi’s evaluation packages, make sure that these are installed in the setup script, which is executed before the job starts (typically during cluster creation).

setup: |
  pip install oumi[evaluation]
Tip

# Custom Evaluations

With Oumi, custom evaluations are effortless and powerful, giving you complete control over how model performance is assessed. Whether you're working with open- or closed-source models, setup is simple: just configure a few settings, no code changes required. Provide your dataset, select your models, and register an evaluation function tailored to the metrics that matter most to you, from accuracy and consistency to bias or domain-specific goals. Oumi handles the rest, including running inference, so you can focus on gaining insights, not managing infrastructure.

## Custom Evaluations Step-by-Step

Running a custom evaluation involves three simple steps. First, define the evaluation configuration using a `YAML` file. Next, register your custom evaluation function to compute the metrics that matter to you. Finally, execute the evaluation using Oumi's {py:class}`~oumi.core.evaluation.Evaluator`, which orchestrates the entire process.

### Step 1: Defining Evaluation Configuration

The evaluation configuration is defined in a `YAML` file and parsed into an {py:class}`~oumi.core.configs.EvaluationConfig` object. Below is a simple example for evaluating GPT-4o. You can evaluate most open models (Llama, DeepSeek, Qwen, Phi, and others), closed models (Gemini, Claude, OpenAI), and cloud-hosted models (Vertex AI, Together, SambaNova, etc.) by simply updating the `model_name` and `inference_engine` fields. Example configurations for popular APIs are available at [Oumi's repo](https://github.com/oumi-ai/oumi/tree/main/configs/apis).

For custom evaluations, always set `evaluation_backend` to `custom`, and assign `task_name` to the name of your registered custom evaluation function (see Step 2). For more details on setting the configuration file for evaluations, including evaluating custom models, refer to our {doc}`documentation </user_guides/evaluate/evaluation_config>`.

```yaml
model:
  model_name: "gpt-4o"

inference_engine: OPENAI

generation:
  max_new_tokens: 8192
  temperature: 0.0

tasks:
  - evaluation_backend: custom
    task_name: my_custom_evaluation
```

### Step 2: Defining Custom Evaluation Function

To define a custom evaluation function, simply register a Python function using the `@register_evaluation_function` decorator. Your function can optionally accept any of the reserved parameters below, depending on your needs:

- `config` ({py:class}`~oumi.core.configs.EvaluationConfig`): The full evaluation configuration defined in Step 1. Include this if you need access to platform-level settings or variables.
- `task_params` ({py:class}`~oumi.core.configs.EvaluationTaskParams`): Represents a specific evaluation task from the `YAML` file. If your configuration defines multiple tasks under `tasks`, this parameter will contain the metadata for the one currently being evaluated.
- `inference_engine` ({py:class}`~oumi.core.inference.BaseInferenceEngine`): An automatically generated engine for the model specified in the evaluation configuration (by `model_name`). Use its {py:obj}`infer() <oumi.core.inference.BaseInferenceEngine.infer>` method to run inference on a list of examples formatted as {class}`~oumi.core.types.conversation.Conversation`.
- User-defined inputs (e.g. `my_input`): You may also include any number of additional parameters of any type. These are passed in during execution (see Step 3).

Your custom evaluation function is expected to return a dictionary where each key is a metric name and each value is the corresponding computed result.

```python
from oumi.core.registry import register_evaluation_function
from oumi.core.configs import EvaluationConfig, EvaluationTaskParams
from oumi.core.inference import BaseInferenceEngine

@register_evaluation_function("my_custom_evaluation")
def my_custom_evaluation(
    config: EvaluationConfig,
    task_params: EvaluationTaskParams,
    inference_engine: BaseInferenceEngine,
    my_input
) -> dict[str, Any]
```

### Step 3: Executing the Evaluation

Once you have defined your `YAML` configuration and registered the custom evaluation function (as specified by the `task_name` in your configuration), you can run the evaluation using the code snippet below.

The {py:class}`~oumi.core.evaluation.Evaluator`'s `evaluate` method requires the evaluation configuration (`config` of type {py:class}`~oumi.core.configs.EvaluationConfig`) to be passed in. It also supports any number of user-defined variables passed as keyword arguments (e.g., `my_input` in the example below). These variable names must exactly match the parameters defined in your custom evaluation function's signature. Otherwise, a runtime error will occur.

The `evaluate` method returns a list of {py:class}`~oumi.core.evaluation.evaluation_result.EvaluationResult` objects, one for each task defined in the `tasks` section of your `YAML` file. Each result includes the dictionary returned by the custom evaluation function (`result.task_result`), along with useful metadata such as `result.start_time`, `result.elapsed_time_sec`, and more.

```python
from oumi.core.configs import EvaluationConfig
from oumi.core.evaluation import Evaluator

config = EvaluationConfig.from_yaml(<path/to/yaml/file>)
results = Evaluator().evaluate(config, my_input=<user_input>)
```

## Walk-through Example

This section walks through a simple example to demonstrate how to use custom evaluations in practice. If you are interested in a more realistic walk-through, see our {gh}`hallucination classifier <notebooks/Oumi - Build your own Custom Evaluation (Hallucination Classifier).ipynb>` notebook.

Suppose you want to assess response verbosity (i.e., the average length of model responses, measured in number of characters) across multiple models. To do this, assume you’ve prepared a dataset of user queries. A toy dataset (`my_conversations`) with two examples is shown below, formatted as a list of {class}`~oumi.core.types.conversation.Conversation` objects.

```python
from oumi.core.types.conversation import Conversation, Message, Role

my_conversations = [
    Conversation(
        messages=[
            Message(role=Role.USER, content="Hello there!"),
        ]
    ),
    Conversation(
        messages=[
            Message(role=Role.USER, content="How are you?"),
        ]
    ),
]
```

### Step 1: Defining the Evaluation Configuration

Start by defining a `YAML` configuration for each model you want to evaluate. The configuration specifies the model, inference engine, and a link to the custom evaluation function via the `task_name`.

```python
gpt_4o_config = """
  model:
    model_name: "gpt-4o"

  inference_engine: OPENAI

  tasks:
  - evaluation_backend: custom
    task_name: model_verboseness_evaluation
"""
```

### Step 2: Defining Custom Evaluation Function

Next, define the evaluation function. Start by using the provided `inference_engine` to run inference and generate model responses. During inference, the engine appends a response (i.e., a {class}`~oumi.core.types.conversation.Message` with role {py:obj}`~oumi.core.types.conversation.Role`=`ASSISTANT`) at the end of each `conversation` (type: {class}`~oumi.core.types.conversation.Conversation`) of the list `conversations`.

You can retrieve the model response from each {class}`~oumi.core.types.conversation.Conversation` using the `last_message()` method, then compute the average character length across all responses, as shown in the example below.

```python
from oumi.core.registry import register_evaluation_function

@register_evaluation_function("model_verboseness_evaluation")
def model_verboseness_evaluation(inference_engine, conversations):
    # Run inference to generate the model responses.
    conversations = inference_engine.infer(conversations)

    aggregate_response_length = 0
    for conversation in conversations:
        # Extract the assistant's (model's) response from the conversation.
        response: str = conversation.last_message().content

        # Update the sum of lengths for all model responses.
        aggregate_response_length += len(response)

    return {"average_response_length": aggregate_response_length / len(conversations)}
```

### Step 3: Executing the Evaluation

Finally, run the evaluation using the code snippet below. This will execute inference and compute the verbosity metric based on your custom evaluation function. Note that `conversations` is a user-defined variable, intended to pass the dataset into the evaluation function.

```python
from oumi.core.configs import EvaluationConfig
from oumi.core.evaluation import Evaluator

config = EvaluationConfig.from_str(gpt_4o_config)
results = Evaluator().evaluate(config, conversations=my_conversations)
```


The average response length can be retrieved from `results` as shown below. Since this walkthrough assumes a single task (defined in the `tasks` section of the `YAML` config), we only examine the first (`[0]`) item in the `results` list.

```python
result_dict = results[0].get_results()
print(f"Average length: {result_dict['average_response_length']}")
```