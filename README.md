# atomic // transfer learning detector

![maintenance-status](https://img.shields.io/badge/maintenance-barely-red.svg)
![coffee-consumed](https://img.shields.io/badge/coffee-fatal_levels-black.svg)
![sanity](https://img.shields.io/badge/sanity-404_not_found-orange.svg)

i wrote this because i got tired of people stealing reinforcement learning models and claiming they "found the same local minima by accident." sure you did, buddy.

this tool compares two rocket league bot models (specifically gigalearn/rlgym-cpp ones) and runs a battery of 8 distinct statistical tests to tell you if bot b is just bot a with a fake mustache.

## 🔧 how it works (simplified)

i drew this instead of sleeping. basically, we load both bots, rip their brains out, and poke them with math until they confess.

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
         │   loading phase (pain)    │
         ▼                           ▼
    load .lt models             load python files
    (the brains)               (the nervous system)
         │                           │
         └─────────────┬─────────────┘
                       │
                       ▼
           ┌───────────────────────┐
           │  the interrogation:   │
           │  1. weight hunting    │
           │  2. activation vibes  │
           │  3. gradient pain     │
           │  4. distribution (kl) │
           │  5. cka (soul check)  │
           │  6. behavior matching │
           │  7. kickoff simul     │
           │  8. spectral entropy  │
           └───────────┬───────────┘
                       │
                       ▼
            ┌──────────────────┐
            │ detection report │
            │ + json results   │
            └──────────────────┘
```

## 📋 configuration files

we switched to json configs because parsing python files with regex was making me lose my will to live. each bot needs a config.

### the schema (don't mess this up)

save this as `bot_config.json`. if you get a syntax error, use a validator. i am not a json debugger.

```json
{
  "bot_name": "MyBot",
  "model_path": "/path/to/folder/with/POLICY.lt",
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
    "action_parser_class": "LookupAction"
  },
  "additional_paths": [
    "/path/to/bot/util",
    "/path/to/bot/src"
  ]
}
```

## 🚀 quick start

### step 1: generate configs automatically

i wrote a script that tries to guess your config. it works 60% of the time, every time.

```bash
# generate config for the victim
python generate_config.py /path/to/BOT1 original.json

# generate config for the suspect
python generate_config.py /path/to/BOT2 sus.json
```

**warning:** open the json files after generating them. if `obs_size` is 0 or `layers` is empty, you need to fix it manually. `generate_config.py` is trying its best, okay?

### step 2: run the detector

this is where we burn the gpu (optional) and catch the thief.

```bash
python tl_detector.py original.json sus.json --device cuda
```

## 🔬 detection methods (the science bit)

we use a point system. max score is like 40 or something. honestly i lost count.

### 1. weight similarity (the smoking gun)
**what it does:** checks if the raw numbers in the neural net are identical.
**interpretation:**
- **>95%**: ctrl+c, ctrl+v. go to jail.
- **<60%**: probably innocent, or they retrained it enough to hide the crime.

### 2. cka (centered kernel alignment)
**what it does:** compares the "internal representation" of data.
**why:** even if they change the layer sizes, cka can sometimes tell if the "knowledge" is the same. it's basically checking if the bots have the same soul.

### 3. distribution similarity (kl divergence)
**what it does:** checks if the bots output the same probabilities for actions.
**why:** if bot A says "jump 90%, boost 10%" and bot B says "jump 89%, boost 11%", that's suspicious.

### 4. kickoff behavior
**what it does:** forces both bots to play 100 kickoffs in a simulation.
**verdict:** if they do the *exact* same pixel-perfect speedflip, that's not a coincidence. that's a copy.

### 5. spectral entropy
**what it does:** checks the complexity of the information flow in the layers.
**why:** copied nets tend to preserve the eigenvalue spectrum even after fine-tuning. don't ask me to explain the linear algebra, i just imported `scipy`.

### 6. transfer learning signature
**what it does:** looks for the "frozen layers" pattern.
**the pattern:** early layers are identical (frozen), later layers are different (fine-tuned). classic lazy developer move.

## 📊 interpreting results

the tool gives you a confidence score. here is the translation guide:

- **0% - 20%**: **innocent**. or they are smarter than me.
- **20% - 50%**: **vague resemblance**. maybe they watched the same youtube tutorial.
- **50% - 85%**: **suspicious**. architectures match, behaviors align, vibes are off.
- **85% - 99%**: **likely transfer**. they took your model and trained it for 10 more minutes.
- **99.9%**: **smoking gun**. exact weight matches found. absolutely shameless.

## ⚠️ troubleshooting

**"model paths do not exist"**
> check your json. `model_path` needs to point to the folder with `POLICY.lt`. not the file itself, the folder. reading is fundamental.

**"could not load components"**
> your python files have imports that `tl_detector` can't find. add the folders to `additional_paths` in the json. or stop writing spaghetti code.

**"shape mismatch warnings"**
> the bots have different architectures. the tool will skip weight comparison and use CKA/Behavior analysis instead. it still works, stop panicking.

**"both bots score high but i know they are different"**
> did you train them on the exact same dataset with the exact same seed? well, there you go.

## license

The "Totally Not Our Fault" License (MIT)
Copyright (c) 2025 Atomic Contributors

Congratulations! You are now in possession of some fine software, absolutely free of charge. You are hereby granted permission to do pretty much whatever you want with it—use it, copy it, modify it, merge it, publish it, distribute it, even sell it (if you can convince someone to pay for it). You can also sublicense it, though we can’t promise anyone will listen.

The only catch? You must include this notice in all copies or substantial portions of the software. You know, so people remember where it came from.

Now, The Serious Bit:
This software is provided "as is," with absolutely no warranties. That means if it breaks, melts your computer, triggers an AI uprising, or somehow causes your cat to ignore you more than usual, we are not responsible.

By using this software, you acknowledge that we, the Atomic Contributors, will not be held liable for any damages, losses, or existential crises that may arise from its use.
