# Transfer Learning Detection System - JSON Configuration Edition

🔍 Comprehensive transfer learning detection for GigaLearnCPP Rocket League bots using JSON configuration files.

## 📋 Table of Contents

- [Quick Start](#quick-start)
- [How It Works](#how-it-works)
- [Configuration Files](#configuration-files)
- [Usage Examples](#usage-examples)
- [Detection Methods](#detection-methods)
- [BOT1 vs BOT2 Example](#BOT1-vs-tromsfr-example)
- [Troubleshooting](#troubleshooting)

## 🚀 Quick Start

### Step 1: Install Dependencies

```bash
pip install torch numpy scipy
```

### Step 2: Generate Bot Configurations

```bash
# Auto-generate config for original bot
python generate_config.py /path/to/BOT1 original.json

# Auto-generate config for suspicious bot
python generate_config.py /path/to/BOT2 sus.json
```

### Step 3: Review and Edit Configs

Open `original.json` and `sus.json` to verify:
- Model paths are correct
- Layer sizes match your bot
- Observation and action parser paths are valid

### Step 4: Run Detection

```bash
python tl_detector.py original.json sus.json
```

## 🔧 How It Works

```
┌─────────────────┐         ┌─────────────────┐
│ original.json   │         │   sus.json      │
│     (BOT1)      │         │    (BOT2)       │
└────────┬────────┘         └────────┬────────┘
         │                           │
         │    ┌──────────────────┐   │
         └───▶│  tl_detector.py  │◀──┘
              └────────┬─────────┘
                       │
         ┌─────────────┴─────────────┐
         │                           │
         ▼                           ▼
    Load Models                 Load Components
    (POLICY.lt,                 (obs.py,
     SHARED_HEAD.lt)             your_act.py)
         │                           │
         └─────────────┬─────────────┘
                       │
                       ▼
           ┌───────────────────────┐
           │  Run 7 Analyses:      │
           │  1. Weight Similarity │
           │  2. Activation Patterns│
           │  3. Gradient Patterns │
           │  4. Distribution (KL) │
           │  5. CKA Alignment     │
           │  6. Behavior Matching │
           │  7. Final Verdict     │
           └───────────┬───────────┘
                       │
                       ▼
            ┌──────────────────┐
            │ Detection Report │
            │ + JSON Results   │
            └──────────────────┘
```

## 📄 Configuration Files

### JSON Schema

Each bot requires a JSON configuration file with this structure:

```json
{
  "bot_name": "BotName",
  "model_path": "/path/to/models",
  "architecture": {
    "shared_head_layers": [512, 512, 512, 512],
    "policy_layers": [1024, 512, 512, 512, 512],
    "activation": "relu",
    "layer_norm": true
  },
  "observation": {
    "obs_size": 109,
    "obs_builder_path": "/path/to/obs.py",
    "obs_builder_class": "AdvancedObs"
  },
  "action_parser": {
    "action_size": 90,
    "action_parser_path": "/path/to/your_act.py",
    "action_parser_class": "LookupAction",
    "action_bins": [[-1, 0, 1], [-1, 0, 1], [-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]
  },
  "discrete_policy": {
    "discrete_policy_path": "/path/to/discrete_policy.py",
    "discrete_policy_class": "DiscreteFF"
  },
  "additional_paths": [
    "/path/to/bot",
    "/path/to/bot/util"
  ],
  "metadata": {
    "tournament": "Tournament Name",
    "author": "Author Name",
    "training_steps": 1000000000,
    "notes": "Additional notes"
  }
}
```

### Field Descriptions

| Field | Required | Description |
|-------|----------|-------------|
| `bot_name` | ✅ | Identifier for the bot |
| `model_path` | ✅ | Directory containing POLICY.lt and SHARED_HEAD.lt |
| `architecture.shared_head_layers` | ✅ | Layer sizes for shared head |
| `architecture.policy_layers` | ✅ | Layer sizes for policy head |
| `observation.obs_size` | ✅ | Size of observation vector |
| `observation.obs_builder_path` | ✅ | Path to obs.py file |
| `action_parser.action_size` | ✅ | Number of discrete actions |
| `action_parser.action_parser_path` | ✅ | Path to action parser file |
| `additional_paths` | ✅ | Python paths needed for imports |

## 📚 Usage Examples

### Example 1: Auto-Generate Configs

```bash
# Generate config for BOT1
python generate_config.py /path/to/BOT1 BOT1.json

# Generate config for BOT2  
python generate_config.py /path/to/BOT2 BOT2.json

# Run detection
python tl_detector.py BOT1.json BOT2.json
```

### Example 2: Manual Config Creation

Create `original.json`:
```json
{
  "bot_name": "MyOriginalBot",
  "model_path": "/home/user/bots/original/models",
  "architecture": {
    "shared_head_layers": [512, 512, 512, 512],
    "policy_layers": [1024, 512, 512, 512, 512]
  },
  "observation": {
    "obs_size": 109,
    "obs_builder_path": "/home/user/bots/original/obs.py",
    "obs_builder_class": "AdvancedObs"
  },
  "action_parser": {
    "action_size": 90,
    "action_parser_path": "/home/user/bots/original/your_act.py",
    "action_parser_class": "LookupAction"
  },
  "additional_paths": [
    "/home/user/bots/original",
    "/home/user/bots/original/util"
  ]
}
```

Run with custom output:
```bash
python tl_detector.py original.json sus.json --output my_results.json
```

### Example 3: GPU Acceleration

```bash
python tl_detector.py original.json sus.json --device cuda
```

## 🔬 Detection Methods

### 1. Weight Similarity Analysis (3 points)

**What it does**: Directly compares neural network weights layer-by-layer

**Metrics**:
- Cosine Similarity: Measures angle between weight vectors
- Pearson Correlation: Linear correlation between weights
- L2 Distance: Euclidean distance between weights

**Interpretation**:
- 🔴 **>0.95**: Near-identical weights (very strong evidence)
- 🟠 **0.8-0.95**: High similarity (likely transfer learning)
- 🟡 **0.6-0.8**: Moderate similarity (possible transfer)
- 🟢 **<0.6**: Low similarity (unlikely transfer)

### 2. Activation Similarity (2 points)

**What it does**: Compares internal representations on random inputs

**Why it's powerful**: Works even with different architectures!

**Metrics**:
- Spearman Correlation: Rank correlation of activations
- Cosine Similarity: Similarity of activation vectors

**Interpretation**:
- 🔴 **>0.7**: Similar internal representations
- 🟡 **0.4-0.7**: Some similarity
- 🟢 **<0.4**: Different representations

### 3. Gradient Similarity (2 points)

**What it does**: Compares gradient patterns during backpropagation

**Why it matters**: Similar gradients suggest similar optimization landscapes (trained from similar starting point)

**Interpretation**:
- 🔴 **>0.6**: Similar gradient patterns
- 🟡 **0.3-0.6**: Some similarity
- 🟢 **<0.3**: Different patterns

### 4. Distribution Similarity - KL Divergence (3 points)

**What it does**: Compares output probability distributions

**Metrics**:
- KL Divergence: Measures how different distributions are
- JS Divergence: Symmetric version of KL

**Why it's critical**: Can detect transfer learning even with different action parsers!

**Interpretation** (lower = more similar):
- 🔴 **<0.1**: Nearly identical distributions
- 🟠 **0.1-0.5**: Very similar
- 🟡 **0.5-1.5**: Somewhat similar
- 🟢 **>1.5**: Different distributions

### 5. CKA - Centered Kernel Alignment (3 points)

**What it does**: State-of-the-art representation similarity metric

**Why it's robust**: 
- Invariant to linear transformations
- Works with different layer sizes
- Published research-backed method

**Interpretation**:
- 🔴 **>0.9**: Nearly identical representations
- 🟠 **0.7-0.9**: Very similar
- 🟡 **0.5-0.7**: Somewhat similar  
- 🟢 **<0.5**: Different

### 6. Behavioral Similarity (2 points)

**What it does**: Uses actual obs builders and action parsers to compare decisions

**Why it's unique**: Tests real in-game behavior, not just neural patterns

**Metrics**:
- Action Agreement Rate: % of times models choose same action
- Probability Correlation: Similarity of action probability distributions

**Interpretation**:
- 🔴 **>0.8**: Almost always agree
- 🟠 **0.6-0.8**: Often agree
- 🟡 **0.4-0.6**: Sometimes agree
- 🟢 **<0.4**: Rarely agree

### 7. Final Verdict (Combined)

**Scoring System**:
- Maximum: 15 points
- Weight Similarity: 3 points
- Activation: 2 points
- Gradient: 2 points
- Distribution: 3 points
- CKA: 3 points
- Behavior: 2 points

**Final Verdicts**:
- 🔴 **80-100%**: HIGHLY LIKELY - Strong evidence across multiple methods
- 🟠 **60-79%**: LIKELY - Significant evidence
- 🟡 **40-59%**: POSSIBLE - Some evidence, investigate further
- 🟢 **20-39%**: UNLIKELY - Little evidence
- 🟢 **0-19%**: NO EVIDENCE - Models appear independent

## 🎯 BOT1 vs BOT2 Example

### Expected Scenario

If BOT2 was transfer-learned from BOT1, you'd expect:

```
Weight Similarity: 🔴 VERY HIGH (0.92-0.98)
├─ Weights are nearly identical
└─ Small variations from fine-tuning

Activation Similarity: 🔴 HIGH (0.75-0.85)
├─ Internal representations very similar
└─ Minor differences in final layers

Gradient Similarity: 🟠 HIGH (0.55-0.70)
├─ Similar optimization landscape
└─ Learned from same initialization

Distribution Similarity: 🔴 VERY HIGH (KL < 0.2)
├─ Output distributions nearly identical
└─ Makes very similar decisions

CKA Similarity: 🔴 VERY HIGH (0.88-0.95)
├─ Learned representations match
└─ Independent of exact architecture

Behavioral Similarity: 🔴 VERY HIGH (0.82-0.92)
├─ Chooses same actions ~85-90% of time
└─ Nearly indistinguishable in-game

FINAL VERDICT: 🔴 HIGHLY LIKELY (85-95%)
```

### Setup for BOT1 vs BOT2

1. **Generate configs**:
```bash
python generate_config.py /path/to/BOT1 BOT1.json
python generate_config.py /path/to/BOT2 tromsfr.json
```

2. **Verify the configs have**:
   - Same architecture (both use [512,512,512,512] and [1024,512,512,512,512])
   - Same obs_size (109)
   - Same action_size (90)
   - Correct paths to their respective files

3. **Run detection**:
```bash
python tl_detector.py BOT1.json tromsfr.json --output tromso_analysis.json
```

4. **Interpret results**:
   - Look for HIGH or VERY HIGH verdicts across multiple categories
   - Check if weight similarity is suspiciously high (>0.9)
   - Examine behavioral agreement rate
   - Review the evidence list in final verdict

## 🔍 Troubleshooting

### "Model paths do not exist"

**Solution**: Check that `model_path` in JSON points to directory with:
- `POLICY.lt` (or `POLICY.LT`)
- `SHARED_HEAD.lt` (or `SHARED_HEAD.LT`)

### "Could not load components"

**Problem**: obs.py or your_act.py not loading properly

**Solutions**:
1. Verify file paths in JSON are absolute paths
2. Check `additional_paths` includes bot directory
3. Ensure no import errors in the Python files
4. Try manually importing the files to test

### "Import error: No module named..."

**Problem**: Missing dependencies from bot's Python files

**Solutions**:
1. Add more paths to `additional_paths` in JSON
2. Install missing packages: `pip install <package>`
3. Check if bot requires util/ subdirectory

### "Shape mismatch" warnings

**Status**: Usually not critical

**Meaning**: Architectures differ slightly, system uses alternative methods

### Low confidence but suspicious behavior

**Actions**:
1. Manually inspect weight files
2. Run with more samples: Edit num_samples in code
3. Check metadata - same training steps?
4. Compare training curves if available

### "Different parameter counts"

**Meaning**: Architectures significantly different

**Impact**: Weight similarity unavailable, but other methods still work

### Both bots score high on everything

**Possible reasons**:
1. Actually transfer learned (valid detection!)
2. Both trained on identical data with identical hyperparameters
3. Converged to same solution (rare)

**Next steps**:
- Check training logs
- Compare training dates/duration
- Ask bot authors directly

## 📊 Output Files

### Detection Results JSON

```json
{
  "original_bot": "BOT1",
  "suspicious_bot": "BOT2",
  "weight_similarity": {
    "avg_cosine_similarity": 0.9234,
    "avg_pearson_correlation": 0.9456,
    "verdict": "🔴 VERY HIGH"
  },
  "activation_similarity": { ... },
  "gradient_similarity": { ... },
  "distribution_similarity": { ... },
  "cka_similarity": { ... },
  "behavior_similarity": { ... },
  "final_verdict": {
    "verdict": "🔴 HIGHLY LIKELY TRANSFER LEARNING DETECTED",
    "confidence_percentage": 86.7,
    "score": 13,
    "max_score": 15,
    "evidence": [
      "Weight Similarity: 🔴 VERY HIGH",
      "Activation Similarity: 🔴 HIGH",
      "Distribution Similarity: 🔴 VERY HIGH",
      "CKA Similarity: 🔴 VERY HIGH",
      "Behavioral Similarity: 🔴 VERY HIGH"
    ]
  }
}
```

## 🎓 Understanding the Science

### Why Multiple Methods?

Different transfer learning techniques are caught by different methods:

| Technique | Best Caught By |
|-----------|----------------|
| Direct Copy | Weight Similarity |
| Fine-tuning | Activation + Weight |
| Knowledge Distillation | Distribution Similarity |
| Feature Extraction | CKA + Activation |
| Different Architectures | CKA + Distribution + Behavior |

### False Positives

Can occur when:
- Both trained on identical setup coincidentally
- Solution space is constrained (simple problem)
- Common preprocessing causes similarity

**Mitigation**: Look for multiple independent sources of evidence

### False Negatives

Can occur when:
- Extensive re-training after transfer
- Major architectural modifications
- Only early layers transferred
- Sophisticated obfuscation

**Mitigation**: Investigate training logs, dates, and other circumstantial evidence

## ⚖️ Legal & Ethical Use

### Appropriate Use
✅ Tournament organizers verifying bot originality  
✅ Investigating suspected rule violations  
✅ Research and education  
✅ Personal verification of your own bots

### Inappropriate Use
❌ Harassing bot creators without evidence  
❌ Making public accusations without full investigation  
❌ Using as sole proof without other evidence  
❌ Violating tournament rules yourself

### Best Practices

1. **Gather multiple evidence sources**: Logs, dates, metadata
2. **Consider confidence levels**: 60% ≠ 100% proof
3. **Give benefit of doubt**: Similar ≠ copied
4. **Follow due process**: Tournament rules and procedures
5. **Be respectful**: Accusations are serious

## 🔗 References

- **GigaLearnCPP**: https://github.com/ZealanL/GigaLearnCPP-Leak
- **CKA Paper**: "Similarity of Neural Network Representations Revisited"
- **Transfer Learning**: "A Survey on Transfer Learning" (Pan & Yang)
- **KL Divergence**: Information theory fundamentals

## 📝 Credits

- Detection system inspired by ML research on model similarity
- Built for RLBot competitive integrity
- Designed for GigaLearnCPP framework

---

**Remember**: This is a detection tool, not a verdict. Always combine with other evidence and follow proper procedures.
