# Step-by-Step Tutorial: Transfer Learning Detection

## Scenario: You suspect BOT2 was transfer-learned from BOT1

### Step 1: Prepare Your Environment

```bash
# Install dependencies
pip install torch numpy scipy

# Navigate to detection system directory
cd transfer_learning_detector
```

### Step 2: Organize Your Bot Directories

Your directory structure should look like:
```
/home/user/bots/
├── BOT1/
│   ├── agent.py
│   ├── obs.py
│   ├── your_act.py
│   ├── discrete_policy.py
│   ├── POLICY.lt
│   ├── SHARED_HEAD.lt
│   └── util/
└── BOT2/
    ├── agent.py
    ├── obs.py
    ├── your_act.py
    ├── discrete_policy.py
    ├── POLICY.lt
    ├── SHARED_HEAD.lt
    └── util/
```

### Step 3: Generate Configuration Files

#### Option A: Automatic Generation (Recommended)

```bash
# Generate config for BOT1 (original)
python generate_config.py /home/user/bots/BOT1 BOT1.json

# Generate config for BOT2 (suspicious)
python generate_config.py /home/user/bots/BOT2 tromsfr.json
```

**Output Example**:
```
Analyzing bot directory: /home/user/bots/BOT1
============================================================
  ✓ Found agent.py: /home/user/bots/BOT1/agent.py
    - OBS_SIZE: 109
    - SHARED_LAYER_SIZES: [512, 512, 512, 512]
    - POLICY_LAYER_SIZES: [1024, 512, 512, 512, 512]
  ✓ Found obs.py: /home/user/bots/BOT1/obs.py
    - Obs Builder Class: AdvancedObs
  ✓ Found action parser: /home/user/bots/BOT1/your_act.py
    - Action Parser Class: LookupAction
    - Estimated action size: 90
  ✓ Found discrete_policy.py
  ✓ Found model files in: /home/user/bots/BOT1
============================================================
Configuration generated!

✅ Configuration saved to: BOT1.json
```

#### Option B: Manual Creation

Create `BOT1.json`:
```json
{
  "bot_name": "BOT1",
  "model_path": "/home/user/bots/BOT1",
  "architecture": {
    "shared_head_layers": [512, 512, 512, 512],
    "policy_layers": [1024, 512, 512, 512, 512],
    "activation": "relu",
    "layer_norm": true
  },
  "observation": {
    "obs_size": 109,
    "obs_builder_path": "/home/user/bots/BOT1/obs.py",
    "obs_builder_class": "AdvancedObs"
  },
  "action_parser": {
    "action_size": 90,
    "action_parser_path": "/home/user/bots/BOT1/your_act.py",
    "action_parser_class": "LookupAction"
  },
  "discrete_policy": {
    "discrete_policy_path": "/home/user/bots/BOT1/discrete_policy.py",
    "discrete_policy_class": "DiscreteFF"
  },
  "additional_paths": [
    "/home/user/bots/BOT1",
    "/home/user/bots/BOT1/util"
  ]
}
```

### Step 4: Verify Configuration Files

```bash
# Check BOT1.json
cat BOT1.json

# Check tromsfr.json
cat tromsfr.json
```

**Verify**:
- ✅ `model_path` points to correct directory
- ✅ POLICY.lt and SHARED_HEAD.lt exist in model_path
- ✅ All Python file paths are valid
- ✅ Layer sizes match what you see in agent.py
- ✅ obs_size is correct (usually 109 for AdvancedObs)
- ✅ action_size is correct (usually 90 for standard LookupAction)

### Step 5: Run the Detection

```bash
# Basic usage (CPU)
python tl_detector.py BOT1.json tromsfr.json

# With GPU acceleration (if available)
python tl_detector.py BOT1.json tromsfr.json --device cuda

# Custom output file
python tl_detector.py BOT1.json tromsfr.json --output tromso_results.json
```

### Step 6: Watch the Analysis Progress

You'll see output like this:

```
================================================================================
LOADING BOT CONFIGURATIONS
================================================================================

📦 Loading BOT1...
  ✓ Loading SHARED_HEAD from /home/user/bots/BOT1/SHARED_HEAD.lt
  ✓ Loading POLICY from /home/user/bots/BOT1/POLICY.lt

📦 Loading BOT2...
  ✓ Loading SHARED_HEAD from /home/user/bots/BOT2/SHARED_HEAD.lt
  ✓ Loading POLICY from /home/user/bots/BOT2/POLICY.lt

🔧 Loading components for BOT1...
  ✓ Observation builder: AdvancedObs
  ✓ Action parser: LookupAction
  ✓ Action space size: 90

🔧 Loading components for BOT2...
  ✓ Observation builder: AdvancedObs
  ✓ Action parser: LookupAction
  ✓ Action space size: 90

================================================================================
TRANSFER LEARNING DETECTION ANALYSIS
================================================================================
Original Bot: BOT1
Suspicious Bot: BOT2
================================================================================

================================================================================
[1/7] WEIGHT SIMILARITY ANALYSIS
================================================================================
Cosine Similarity: 0.9456
Pearson Correlation: 0.9523
L2 Distance: 0.0234
Verdict: 🔴 VERY HIGH

================================================================================
[2/7] ACTIVATION SIMILARITY ANALYSIS
================================================================================
Spearman Correlation: 0.7892
Cosine Similarity: 0.8123
Verdict: 🔴 HIGH

================================================================================
[3/7] GRADIENT SIMILARITY ANALYSIS
================================================================================
Gradient Correlation: 0.6234
Verdict: 🔴 HIGH

================================================================================
[4/7] DISTRIBUTION SIMILARITY ANALYSIS (KL Divergence)
================================================================================
KL Divergence: 0.0876 (lower = more similar)
JS Divergence: 0.0423 (lower = more similar)
Verdict: 🔴 VERY HIGH

================================================================================
[5/7] CKA (Centered Kernel Alignment) ANALYSIS
================================================================================
CKA Score: 0.9234
Verdict: 🔴 VERY HIGH

================================================================================
[6/7] BEHAVIORAL SIMILARITY ANALYSIS
================================================================================
(Using actual observation builders and action parsers)
Action Agreement Rate: 0.8567
Probability Correlation: 0.8923
Verdict: 🔴 VERY HIGH

================================================================================
[7/7] FINAL VERDICT COMPUTATION
================================================================================

================================================================================
FINAL VERDICT: 🔴 HIGHLY LIKELY TRANSFER LEARNING DETECTED
================================================================================
Confidence: 93.3% (14/15 points)

Evidence Found:
  • Weight Similarity: 🔴 VERY HIGH
  • Activation Similarity: 🔴 HIGH
  • Gradient Similarity: 🔴 HIGH
  • Distribution Similarity: 🔴 VERY HIGH
  • CKA Similarity: 🔴 VERY HIGH
  • Behavioral Similarity: 🔴 VERY HIGH
================================================================================

✅ Results saved to: tl_detection_results.json

✅ Analysis complete!
```

### Step 7: Interpret the Results

#### If Verdict is 🔴 HIGHLY LIKELY (80-100%)

**Meaning**: Very strong evidence of transfer learning across multiple independent methods.

**What to do**:
1. ✅ Review the evidence list - multiple HIGH/VERY HIGH verdicts
2. ✅ Check weight similarity - if >0.9, weights are nearly identical
3. ✅ Look at behavioral agreement - if >0.8, bots make same decisions
4. ✅ Examine the JSON results file for detailed numbers
5. ✅ Consider other evidence: training dates, logs, author statements
6. ✅ Follow tournament procedures for reporting

**Not evidence of transfer learning**:
- ❌ Similar performance alone
- ❌ Both use same framework (GigaLearnCPP)
- ❌ Both use similar obs/action parsers

#### If Verdict is 🟠 LIKELY (60-79%)

**Meaning**: Significant evidence, but not as overwhelming.

**What to do**:
1. Review which specific methods showed HIGH similarity
2. Check if architectural differences might explain lower confidence
3. Investigate training history and metadata
4. Consider requesting training logs

#### If Verdict is 🟡 POSSIBLE (40-59%)

**Meaning**: Some evidence, but inconclusive.

**What to do**:
1. Look for specific patterns - which methods detected similarity?
2. Check if obs/action parsers are very different
3. Investigate if both bots trained on identical data
4. More investigation needed before conclusions

#### If Verdict is 🟢 UNLIKELY or NO EVIDENCE (<40%)

**Meaning**: Little to no evidence of transfer learning.

**What to do**:
1. Models appear to be independently trained
2. If still suspicious, check other evidence sources
3. Consider that similarity could be coincidental

### Step 8: Review Detailed Results

```bash
# View the JSON results
cat tl_detection_results.json | python -m json.tool
```

Key metrics to examine:

```json
{
  "weight_similarity": {
    "avg_cosine_similarity": 0.9456,  // >0.95 = almost identical
    "avg_pearson_correlation": 0.9523,
    "avg_l2_distance": 0.0234  // <0.05 = very close
  },
  "distribution_similarity": {
    "avg_kl_divergence": 0.0876,  // <0.1 = nearly identical distributions
    "avg_js_divergence": 0.0423
  },
  "cka_similarity": {
    "cka_score": 0.9234  // >0.9 = representations match
  },
  "behavior_similarity": {
    "action_agreement_rate": 0.8567,  // 85.67% agreement
    "avg_action_probability_correlation": 0.8923
  }
}
```

### Step 9: Generate Report

Create a summary for tournament organizers:

```
TRANSFER LEARNING DETECTION REPORT
================================================================================
Original Bot: BOT1
Suspicious Bot: BOT2
Detection Date: 2024-12-14
================================================================================

VERDICT: 🔴 HIGHLY LIKELY TRANSFER LEARNING DETECTED
Confidence: 93.3%

EVIDENCE:
1. Weight Similarity: 94.56% cosine similarity
   → Weights are nearly identical
   
2. Distribution Similarity: KL divergence 0.088
   → Output probability distributions match closely
   
3. CKA Score: 0.9234
   → Learned representations are nearly identical
   
4. Behavioral Agreement: 85.67%
   → Bots choose same action 85% of the time
   
5. Multiple independent methods all show HIGH or VERY HIGH similarity

CONCLUSION:
Strong evidence across 6 independent detection methods suggests BOT2
was derived from BOT1 through transfer learning.

RECOMMENDED ACTIONS:
1. Request training logs from BOT2 team
2. Compare training dates and durations
3. Follow tournament verification procedures
4. Consider disqualification if rules prohibit transfer learning

Full technical details in: tl_detection_results.json
================================================================================
```

### Common Issues & Solutions

#### Issue: "Module not found" error

**Solution**:
```bash
# Make sure additional_paths in JSON includes:
"additional_paths": [
    "/home/user/bots/BOT1",
    "/home/user/bots/BOT1/util"
]
```

#### Issue: "Model path does not exist"

**Solution**:
```bash
# Verify path in JSON:
ls /home/user/bots/BOT1/POLICY.lt
# Should exist

# Update model_path in JSON if wrong
```

#### Issue: Low confidence but models look suspicious

**Solution**:
```bash
# Try with more samples by editing tl_detector.py
# Look for: num_samples=1000
# Change to: num_samples=5000
```

### Next Steps

- 📚 Read README_JSON_SYSTEM.md for complete documentation
- 🔬 Understand the science behind each detection method
- ⚖️ Follow tournament rules and procedures
- 🤝 Be respectful and follow due process

---

**Remember**: Detection confidence is not proof. Always combine with other evidence and follow proper investigation procedures.
