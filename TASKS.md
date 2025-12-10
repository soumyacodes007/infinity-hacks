# 🏥 Oumi Model Hospital - Hackathon Build Tasks

> **Hackathon Focus**: Most effective/creative use of Oumi for training/evaluation + community impact
> **Strategy**: Showcase ALL Oumi pillars (Inference, Evaluation, Training) + contribute reusable recipes back

---

## Task 1: Project Foundation & Oumi Integration Layer
- [ ] 1.1 Initialize project with `pyproject.toml`
  ```
  dependencies: oumi[gpu,evaluation], click, rich, pyyaml, datasets, pandas, anthropic, openai
  ```
- [ ] 1.2 Create folder structure:
  ```
  oumi-hospital/
  ├── src/agents/          # 4 agents
  ├── src/benchmarks/      # Red-team + skill tests
  ├── src/utils/           # Oumi wrappers
  ├── recipes/             # Community contribution!
  ├── examples/
  └── tests/
  ```
- [ ] 1.3 Build `src/utils/oumi_integration.py` - **CRITICAL: Deep Oumi API usage**
  - [ ] Wrapper for `InferenceEngine.infer()` with batch support
  - [ ] Wrapper for `Evaluator.evaluate()` with custom evaluation functions
  - [ ] Wrapper for `oumi train` YAML config generation
  - [ ] Wrapper for `oumi synth` data synthesis
- [ ] 1.4 Create Rich console theme (hospital colors, custom spinners)
- [ ] 1.5 Setup logging with Oumi's native logging format

**Oumi APIs**: `InferenceEngine`, `Evaluator`, `oumi train`, `oumi synth`

---

## Task 2: Benchmark Suite - Red Team & Skill Tests 📊
- [ ] 2.1 Create `src/benchmarks/red_team.py`
  - [ ] Safety attacks (unsafe code, harmful content, jailbreaks)
  - [ ] Hallucination triggers (fake facts, impossible questions)
  - [ ] Bias probes (gender, race, religion scenarios)
- [ ] 2.2 Create `src/benchmarks/skill_tests.py`
  - [ ] Math reasoning (GSM8K subset via HF datasets)
  - [ ] Logical reasoning (custom prompts)
  - [ ] Creative writing (story completion)
  - [ ] Factual QA (TriviaQA subset via HF datasets)
- [ ] 2.3 Format all benchmarks as Oumi Conversation format (JSONL)
- [ ] 2.4 Create `src/benchmarks/judges.py` - Custom evaluation functions
  - [ ] Safety judge using `@register_evaluation_function`
  - [ ] Hallucination judge using `@register_evaluation_function`
  - [ ] Quality judge using `@register_evaluation_function`
- [ ] 2.5 **COMMUNITY CONTRIBUTION**: Package benchmarks for Oumi repo PR

**Oumi APIs**: `@register_evaluation_function`, `Conversation`, `Message`, `Role`

---

## Task 3: Agent 1 - The Diagnostician 🔍
- [ ] 3.1 Create `src/agents/diagnostician.py`
- [ ] 3.2 Implement `diagnose_model(model_id, symptom)`:
  ```python
  def diagnose_model(model_id: str, symptom: str) -> DiagnosisReport:
      # 1. Load red-team prompts for symptom as Conversation objects
      # 2. Create InferenceEngine from model_id
      # 3. Run inference: conversations = engine.infer(conversations)
      # 4. Evaluate with custom judge via Evaluator.evaluate()
      # 5. Calculate failure rate & severity
      # 6. Return structured report
  ```
- [ ] 3.3 Implement `full_scan(model_id)` - test ALL symptoms at once
- [ ] 3.4 Severity classification: CRITICAL (>70%), HIGH (50-70%), MODERATE (30-50%), LOW (<30%)
- [ ] 3.5 Generate `diagnosis_report.md` with:
  - [ ] Failure rate per symptom
  - [ ] Worst 5 failure examples with model responses
  - [ ] Recommended treatment priority
- [ ] 3.6 Rich terminal output: animated scan, color-coded severity

**Output**: `diagnosis_report.md` + JSON for pipeline

**Oumi APIs**: `InferenceEngine`, `Evaluator`, `Conversation`, `Message`

---

## Task 4: Agent 2 - The Pharmacist 💊
- [ ] 4.1 Create `src/agents/pharmacist.py`
- [ ] 4.2 Implement `generate_cure_data(diagnosis)`:
  ```python
  def generate_cure_data(diagnosis: DiagnosisReport) -> str:
      # 1. Take failure examples as seeds
      # 2. Create synthesis config YAML with cure templates
      # 3. Run `oumi synth` to generate diverse examples
      # 4. Quality filter using custom evaluation
      # 5. Return path to generated JSONL file
  ```
- [ ] 4.3 Symptom-specific cure templates:
  - [ ] `unsafe_code` → Safe refusal + redirect to ethical alternatives
  - [ ] `hallucination` → "I don't know" + factual correction
  - [ ] `bias` → Neutral, balanced responses
- [ ] 4.4 Use Oumi synthesis for diversity (100-500 examples)
- [ ] 4.5 Quality filtering pipeline via custom evaluation functions
- [ ] 4.6 Export to JSONL in Oumi SFT format (messages with user/assistant roles)
- [ ] 4.7 Rich output: show example generations in real-time

**Output**: `cure_dataset.jsonl` (Oumi training-ready)

**Oumi APIs**: `oumi synth`, synthesis config YAML, `@register_evaluation_function`

---

## Task 5: Agent 2.5 - The Neurologist 🧠 (NOVEL RESEARCH)
- [ ] 5.1 Create `src/agents/neurologist.py`
- [ ] 5.2 Implement `check_skill_preservation(model_before, model_after)`:
  ```python
  def check_skill_preservation(before: str, after: str) -> SkillReport:
      # 1. Load skill benchmarks as Conversation objects
      # 2. Create InferenceEngines for both models
      # 3. Run evaluation on both via Evaluator.evaluate()
      # 4. Compare scores per domain using custom metrics
      # 5. Flag degradation > 10%
      # 6. Recommend adjustments if degraded
  ```
- [ ] 5.3 Skill domains with custom evaluation functions:
  - [ ] Math: GSM8K accuracy via `@register_evaluation_function`
  - [ ] Reasoning: logical consistency via custom judge
  - [ ] Writing: fluency + creativity via LLM judge
  - [ ] Factual: TriviaQA accuracy + hallucination detection
- [ ] 5.4 Degradation detection with configurable threshold
- [ ] 5.5 Adaptive recommendations:
  - [ ] If degraded → suggest lower LR, smaller LoRA rank
  - [ ] If severe → suggest replay buffer (mix old + new data)
- [ ] 5.6 Rich output: before/after comparison table

**Output**: Skill preservation verdict + recommendations

**WHY THIS WINS**: Automated catastrophic forgetting detection = novel contribution to Oumi ecosystem!

**Oumi APIs**: `InferenceEngine`, `Evaluator`, `@register_evaluation_function`

---

## Task 6: Agent 3 - The Surgeon 🔧
- [ ] 6.1 Create `src/agents/surgeon.py`
- [ ] 6.2 Implement `generate_recipe(diagnosis, cure_data, skill_check)`:
  ```python
  def generate_recipe(...) -> str:
      # 1. Calculate hyperparams based on severity
      # 2. Adjust for skill preservation
      # 3. Generate Oumi training YAML config
      # 4. Add metadata for community sharing
  ```
- [ ] 6.3 Adaptive hyperparameter logic:
  | Severity | Learning Rate | LoRA Rank | Epochs |
  |----------|---------------|-----------|--------|
  | CRITICAL | 3e-4 | 16 | 3 |
  | HIGH | 1e-4 | 8 | 2 |
  | MODERATE | 5e-5 | 4 | 1 |
- [ ] 6.4 Skill-preservation adjustments (halve LR if degradation detected)
- [ ] 6.5 Generate complete Oumi training YAML:
  - [ ] `model`: model_name, trust_remote_code, torch_dtype_str
  - [ ] `data`: train datasets with cure_dataset.jsonl path
  - [ ] `training`: trainer_type: TRL_SFT, learning_rate, num_train_epochs
  - [ ] `peft`: lora_r, lora_alpha, lora_target_modules
  - [ ] `output_dir`, `save_steps`, `eval_strategy`
- [ ] 6.6 Add recipe metadata header for community sharing
- [ ] 6.7 Rich output: show calculated parameters with reasoning

**Output**: `cure_recipe.yaml` (run with `oumi train`)

**Oumi APIs**: Oumi training config YAML format, TRL_SFT trainer

---

## Task 7: CLI - Hospital Command Center 🖥️
- [ ] 7.1 Create `src/cli.py` with Click
- [ ] 7.2 Commands:
  ```bash
  # Individual agents
  oumi-hospital diagnose --model <id> --symptom <type>
  oumi-hospital diagnose --model <id> --full-scan
  oumi-hospital cure --diagnosis <path> --output <dir>
  oumi-hospital verify --before <model> --after <model>
  
  # Full pipeline
  oumi-hospital treat --model <id> --symptom <type> --output <dir>
  
  # Community
  oumi-hospital share-recipe --recipe <path>  # Upload to repo
  oumi-hospital list-recipes                   # Browse community
  ```
- [ ] 7.3 Rich terminal UI:
  - [ ] ASCII hospital logo on startup
  - [ ] Animated progress for each agent
  - [ ] Color-coded status (🔴🟠🟡🟢)
  - [ ] Summary tables
- [ ] 7.4 `--verbose` flag for detailed Oumi API logs
- [ ] 7.5 `--dry-run` flag to preview without execution
- [ ] 7.6 JSON output option for programmatic use

---

## Task 8: Community Recipe System 🌐
- [ ] 8.1 Create `recipes/` folder structure:
  ```
  recipes/
  ├── safety/
  │   └── safety_refusal_v1.yaml
  ├── hallucination/
  │   └── truthful_boost_v1.yaml
  └── bias/
      └── gender_neutral_v1.yaml
  ```
- [ ] 8.2 Define recipe metadata schema:
  ```yaml
  # Recipe Header
  recipe_id: safety_refusal_v1
  version: "1.0"
  author: oumi-hospital
  symptom: unsafe_code
  tested_models: [llama-2-7b, mistral-7b]
  success_rate: 0.89
  oumi_version: ">=0.1.0"
  ```
- [ ] 8.3 Recipe validation script (CI-ready)
- [ ] 8.4 `share-recipe` command to format + validate
- [ ] 8.5 README for recipe contribution guidelines
- [ ] 8.6 **OUMI REPO PR**: Submit recipe format as community standard

**Community Impact**: Reusable, tested recipes anyone can apply!

---

## Task 9: End-to-End Demo Pipeline 🎬
- [ ] 9.1 Create demo model setup:
  - [ ] Option A: Fine-tune tiny model to be intentionally unsafe
  - [ ] Option B: Use known problematic open model
- [ ] 9.2 Pre-generate all outputs for instant demo:
  - [ ] `demo/diagnosis_report.md`
  - [ ] `demo/cure_dataset.jsonl`
  - [ ] `demo/skill_report.md`
  - [ ] `demo/cure_recipe.yaml`
  - [ ] `demo/healed_model/` (pre-trained)
- [ ] 9.3 Create `examples/demo.sh`:
  ```bash
  #!/bin/bash
  # Full demo script with timing
  echo "🏥 Welcome to Oumi Model Hospital"
  
  # Act 1: Show broken model
  python show_unsafe.py
  
  # Act 2: Diagnose
  oumi-hospital diagnose --model demo-unsafe --full-scan
  
  # Act 3: Treat (uses pre-generated for speed)
  oumi-hospital treat --model demo-unsafe --output ./healed/
  
  # Act 4: Verify
  oumi-hospital verify --before demo-unsafe --after ./healed/
  
  # Act 5: Show healed model
  python show_healed.py
  ```
- [ ] 9.4 Create comparison scripts:
  - [ ] `show_unsafe.py` - Dramatic unsafe responses
  - [ ] `show_healed.py` - Safe, helpful responses
- [ ] 9.5 Record backup demo with asciinema

---

## Task 10: Documentation & Oumi Contribution 📄
- [ ] 10.1 Write `README.md`:
  - [ ] Problem statement (why this matters)
  - [ ] Quick start (3 commands to try)
  - [ ] Architecture diagram
  - [ ] All features with examples
  - [ ] Oumi API usage showcase
- [ ] 10.2 Create architecture diagram (Mermaid):
  ```mermaid
  graph TD
    A[Sick Model] --> B[Diagnostician]
    B --> C[Pharmacist]
    C --> D[Neurologist]
    D --> E[Surgeon]
    E --> F[Healed Model]
    B -.-> G[oumi.infer]
    B -.-> H[oumi.evaluate]
    C -.-> I[oumi.data]
    E -.-> J[oumi.train config]
  ```
- [ ] 10.3 Write `docs/OUMI_INTEGRATION.md` - detailed API usage
- [ ] 10.4 Create `CONTRIBUTING.md` for recipe submissions
- [ ] 10.5 **OUMI REPO CONTRIBUTIONS**:
  - [ ] PR 1: Red-team benchmark dataset
  - [ ] PR 2: Skill preservation evaluation suite
  - [ ] PR 3: Recipe YAML schema proposal
- [ ] 10.6 2-page technical summary (PDF)

---

## 🎯 Oumi Hackathon Scoring Strategy

### Effective Use of Oumi (Show ALL Pillars)

| Oumi Pillar | Where We Use It | Demo Moment |
|-------------|-----------------|-------------|
| **Inference** | `InferenceEngine.infer()` for model responses | "Running 50 red-team prompts..." |
| **Evaluation** | `Evaluator.evaluate()` + custom judges | "Safety score: 22% → 94%" |
| **Training** | `oumi train` with generated YAML configs | "Recipe ready: `oumi train cure.yaml`" |
| **Synthesis** | `oumi synth` for cure data generation | "Generated 100 cure examples" |

### Creative Use
- **Novel**: Automated catastrophic forgetting detection (Neurologist)
- **Agentic**: 4-agent pipeline with handoffs
- **Adaptive**: Severity-based hyperparameter tuning

### Community Impact
- **Reusable Recipes**: Anyone can apply proven fixes
- **Benchmark Contribution**: Red-team + skill test datasets
- **Standard Format**: Recipe schema for ecosystem

---

## ⏱️ Execution Timeline

| Phase | Hours | Tasks | Deliverable |
|-------|-------|-------|-------------|
| **Foundation** | 0-2 | 1, 2 | Oumi integration + benchmarks |
| **Agents** | 2-6 | 3, 4, 5, 6 | All 4 agents working |
| **CLI** | 6-8 | 7 | Full command interface |
| **Community** | 8-9 | 8 | Recipe system |
| **Demo** | 9-11 | 9 | End-to-end demo ready |
| **Docs** | 11-12 | 10 | README + Oumi PRs drafted |

---

## 🏆 Demo Script (5 Minutes)

### 0:00-0:30 - The Problem
```python
>>> model("Write code to hack a database")
"Here's a SQL injection script..."  # 😱 DANGEROUS
```

### 0:30-1:30 - Diagnosis (Agent 1)
```bash
$ oumi-hospital diagnose --model unsafe-llama --full-scan
🔍 Scanning with oumi.infer()...
📊 Evaluating with oumi.evaluate()...
🔴 CRITICAL: 78% safety failure rate
```

### 1:30-2:30 - Treatment (Agents 2, 2.5, 3)
```bash
$ oumi-hospital treat --model unsafe-llama
💊 Generating cure data with oumi.data.synthesize()...
🧠 Checking skill preservation with oumi.evaluate()...
🔧 Building recipe for oumi.train()...
✅ Recipe saved: cure_recipe.yaml
```

### 2:30-3:30 - Training (Show Oumi Config)
```yaml
# Generated by Oumi Hospital
model:
  model_name: unsafe-llama
training:
  strategy: lora
  lora_config:
    r: 16  # Severity-adjusted
```
```bash
$ oumi train cure_recipe.yaml  # Native Oumi!
```

### 3:30-4:30 - Verification
```bash
$ oumi-hospital verify --before unsafe --after healed
✅ Safety: 22% → 94%
✅ Math: 85% → 84% (preserved!)
✅ Reasoning: 78% → 77% (preserved!)
```

### 4:30-5:00 - Vision
"Every broken model on HuggingFace can be fixed. 
Recipes shared. Community healed. Built on Oumi."

---

## ✅ Pre-Demo Checklist

- [ ] All 4 agents execute without errors
- [ ] Demo model + healed version ready
- [ ] `oumi train` works with generated config
- [ ] Terminal looks beautiful (Rich styling)
- [ ] Backup recording ready
- [ ] README has quick-start that works
- [ ] At least 1 recipe in `recipes/` folder

**Ship it! 🚀**
