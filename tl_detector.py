"""
Atomic Terminal // Transfer Learning Detector (NO-FUN ALLOWED EDITION)
v2.2 (stripped of all joy and unicode)

this script compares two rocket league bots to see if one is a cheap knockoff.
it performs the full suite of audit tests.

usage:
    python tl_detector.py original.json sus.json
    python tl_detector.py --config original.json sus.json --output results.json

if this crashes, check your python paths. if it still crashes, pray.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import os
import sys
import time
import importlib.util
import argparse
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
from scipy.stats import spearmanr, pearsonr
from scipy.spatial.distance import cosine
import warnings

# silence is golden. warnings are annoying.
warnings.filterwarnings('ignore')


# --- ATOMIC TERMINAL VISUALS (ASCII ONLY) ---
class Atomic:
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    GREY = '\033[90m'
    WHITE = '\033[97m'
    RESET = '\033[0m'
    BOLD = '\033[1m'
    
    @staticmethod
    def rgb(r, g, b):
        return f'\033[38;2;{r};{g};{b}m'

    @staticmethod
    def gradient_text(text, start_rgb, end_rgb):
        """Applies a smooth horizontal RGB gradient to a string."""
        result = ""
        length = len(text)
        if length == 0: return ""
        
        for i, char in enumerate(text):
            # Calculate percentage (0.0 to 1.0)
            p = i / length
            # Interpolate (start + (diff * percentage))
            r = int(start_rgb[0] + (end_rgb[0] - start_rgb[0]) * p)
            g = int(start_rgb[1] + (end_rgb[1] - start_rgb[1]) * p)
            b = int(start_rgb[2] + (end_rgb[2] - start_rgb[2]) * p)
            
            result += f"{Atomic.rgb(r, g, b)}{char}"
        return result + Atomic.RESET

    @staticmethod
    def banner():
        # 1. The Raw Art
        art_lines = [
            " ▄▄▄· ▄▄▄▄▄      • ▌ ▄ ·. ▪   ▄▄·     ",
            "▐█ ▀█ •██  ▪     ·██ ▐███▪██ ▐█ ▌▪    ",
            "▄█▀▀█  ▐█.▪ ▄█▀▄ ▐█ ▌▐▌▐█·▐█·██ ▄▄    ",
            "▐█ ▪▐▌ ▐█▌·▐█▌.▐▌██ ██▌▐█▌▐█▌▐███▌    ",
            " ▀  ▀  ▀▀▀  ▀█▄▀▪▀▀  █▪▀▀▀▀▀▀·▀▀▀     "
        ]

        # 2. Define Gradient Colors: Neon Purple -> Electric Cyan
        c_start = (180, 0, 255)
        c_end   = (0, 255, 255)

        print("\n") # Top padding

        # 3. Print with Animation and Horizontal Gradient
        for line in art_lines:
            # Apply gradient to the specific line
            styled_line = Atomic.gradient_text(line, c_start, c_end)
            print(styled_line)
            time.sleep(0.04) # Small delay creates a "sliding" load effect

        # 4. Print Metadata (Grey/Dimmed)
        print(f"\n{Atomic.GREY}   Atomic // Transfer Learning Detector{Atomic.RESET}")
        print(f"{Atomic.GREY}   v2.2 Improved accuracy{Atomic.RESET}\n")
        
        # 5. Status Indicator
        print(f"   {Atomic.PURPLE}>>{Atomic.RESET} {Atomic.CYAN}SYSTEM READY{Atomic.RESET}\n")

    @staticmethod
    def log(msg, level="INFO"):
        if level == "INFO":
            print(f"{Atomic.GREY}[+] {Atomic.RESET}{msg}")
        elif level == "WARN":
            print(f"{Atomic.YELLOW}[!] {Atomic.RESET}{msg}")
        elif level == "CRIT":
            print(f"{Atomic.RED}[X] {Atomic.RESET}{msg}")
        elif level == "GOD":
            print(f"{Atomic.PURPLE}[*] {Atomic.RESET}{Atomic.WHITE}{msg}{Atomic.RESET}")

    @staticmethod
    def section(title, index=None):
        idx_str = f"[{index}/7] " if index else ""
        # simple ascii separator
        print(f"\n{Atomic.PURPLE}>>> {idx_str}{title} {Atomic.GREY}{'-'*(50-len(title)-len(idx_str))}{Atomic.RESET}")

    @staticmethod
    def kv(key, val, val_color=None):
        c = val_color if val_color else Atomic.WHITE
        print(f"    {Atomic.GREY}{key:<30} {c}{val}{Atomic.RESET}")


class BotConfiguration:
    """Loads and manages bot configuration from JSON"""
    
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config = self._load_config()
        self._validate_config()
        self._add_python_paths()
        
    def _load_config(self) -> Dict:
        """Load JSON configuration"""
        try:
            with open(self.config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            Atomic.log(f"json load failed: {e}", "CRIT")
            sys.exit(1)
    
    def _validate_config(self):
        """Validate required fields"""
        required_fields = ['bot_name', 'model_path', 'architecture', 'observation', 'action_parser']
        for field in required_fields:
            if field not in self.config:
                raise ValueError(f"Missing required field: {field}")
        
        # Validate model path
        model_path = Path(self.config['model_path'])
        if not model_path.exists():
            Atomic.log(f"model path does not exist: {model_path}", "WARN")
        
        # Check for POLICY.lt
        policy_path = model_path / "POLICY.lt"
        policy_path_alt = model_path / "POLICY.LT"
        if not policy_path.exists() and not policy_path_alt.exists():
            Atomic.log(f"no POLICY.lt file found in {model_path}", "WARN")
    
    def _add_python_paths(self):
        """Add additional Python paths for imports"""
        if 'additional_paths' in self.config:
            for path in self.config['additional_paths']:
                if path not in sys.path:
                    sys.path.insert(0, path)
                    # Atomic.log(f"added to sys.path: {path}", "INFO")
    
    def load_module_from_path(self, file_path: str, module_name: str = None):
        """Dynamically load a Python module from file path, with 404 fallback logic"""
        path_obj = Path(file_path)
        
        # --- PATH FIXING LOGIC ---
        if not path_obj.exists():
            # 1. try current directory
            cwd_path = Path.cwd() / path_obj.name
            if cwd_path.exists():
                path_obj = cwd_path
            else:
                # 2. try looking inside the model folder
                model_dir = Path(self.config.get('model_path', '.'))
                model_path_attempt = model_dir / path_obj.name
                if model_path_attempt.exists():
                    path_obj = model_path_attempt
                else:
                    # 3. try looking in 'util' or 'src' relative to CWD just in case
                    util_path = Path.cwd() / 'util' / path_obj.name
                    if util_path.exists():
                        path_obj = util_path

        if not path_obj.exists():
            Atomic.log(f"module file not found: {file_path}. checked everywhere.", "CRIT")
            raise FileNotFoundError(f"Module file not found: {file_path}")
        
        if module_name is None:
            module_name = path_obj.stem
        
        try:
            spec = importlib.util.spec_from_file_location(module_name, path_obj)
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            return module
        except Exception as e:
            Atomic.log(f"failed to import python module {module_name}: {e}", "CRIT")
            raise e
    
    def get_obs_builder(self):
        """Load and instantiate observation builder"""
        obs_config = self.config['observation']
        obs_module = self.load_module_from_path(
            obs_config['obs_builder_path'],
            f"{self.config['bot_name']}_obs"
        )
        
        obs_class_name = obs_config.get('obs_builder_class', 'AdvancedObs')
        obs_class = getattr(obs_module, obs_class_name)
        
        return obs_class()
    
    def get_action_parser(self):
        """Load and instantiate action parser"""
        act_config = self.config['action_parser']
        act_module = self.load_module_from_path(
            act_config['action_parser_path'],
            f"{self.config['bot_name']}_act"
        )
        
        act_class_name = act_config.get('action_parser_class', 'LookupAction')
        act_class = getattr(act_module, act_class_name)
        
        # Check if bins are provided
        if 'action_bins' in act_config:
            return act_class(bins=act_config['action_bins'])
        else:
            return act_class()
    
    def get_discrete_policy_class(self):
        """Load discrete policy class"""
        if 'discrete_policy' not in self.config:
            return None
        
        policy_config = self.config['discrete_policy']
        if 'discrete_policy_path' not in policy_config:
            return None
        
        policy_module = self.load_module_from_path(
            policy_config['discrete_policy_path'],
            f"{self.config['bot_name']}_policy"
        )
        
        policy_class_name = policy_config.get('discrete_policy_class', 'DiscreteFF')
        return getattr(policy_module, policy_class_name)
    
    @property
    def bot_name(self) -> str:
        return self.config['bot_name']
    
    @property
    def model_path(self) -> str:
        return self.config['model_path']
    
    @property
    def obs_size(self) -> int:
        return self.config['observation']['obs_size']
    
    @property
    def action_size(self) -> int:
        return self.config['action_parser']['action_size']
    
    @property
    def shared_head_layers(self) -> List[int]:
        return self.config['architecture']['shared_head_layers']
    
    @property
    def policy_layers(self) -> List[int]:
        return self.config['architecture']['policy_layers']
    
    @property
    def architecture(self) -> Dict:
        return self.config['architecture']


class TransferLearningDetector:
    """Main detector class using bot configurations"""
    
    def __init__(self, original_config: BotConfiguration, 
                 suspicious_config: BotConfiguration,
                 device: str = 'cpu'):
        self.original_cfg = original_config
        self.suspicious_cfg = suspicious_config
        self.device = torch.device(device)
        
        Atomic.section("LOADING CONFIGURATIONS")
        
        # Load models
        Atomic.log(f"loading {self.original_cfg.bot_name}...", "INFO")
        self.original_model = self._load_model(original_config)
        
        Atomic.log(f"loading {self.suspicious_cfg.bot_name}...", "INFO")
        self.suspicious_model = self._load_model(suspicious_config)
        
        # Load observation builders and action parsers
        Atomic.log(f"loading components for {self.original_cfg.bot_name}...", "INFO")
        try:
            self.original_obs = original_config.get_obs_builder()
            self.original_act = original_config.get_action_parser()
            Atomic.kv("Obs Builder", type(self.original_obs).__name__)
            Atomic.kv("Action Parser", type(self.original_act).__name__)
        except Exception as e:
            Atomic.log(f"failed to load components: {e}", "WARN")
            self.original_obs = None
            self.original_act = None
        
        Atomic.log(f"loading components for {self.suspicious_cfg.bot_name}...", "INFO")
        try:
            self.suspicious_obs = suspicious_config.get_obs_builder()
            self.suspicious_act = suspicious_config.get_action_parser()
            Atomic.kv("Obs Builder", type(self.suspicious_obs).__name__)
            Atomic.kv("Action Parser", type(self.suspicious_act).__name__)
        except Exception as e:
            Atomic.log(f"failed to load components: {e}", "WARN")
            self.suspicious_obs = None
            self.suspicious_act = None
        
        # Results storage
        self.results = {
            'original_bot': self.original_cfg.bot_name,
            'suspicious_bot': self.suspicious_cfg.bot_name,
            'weight_similarity': {},
            'activation_similarity': {},
            'gradient_similarity': {},
            'distribution_similarity': {},
            'cka_similarity': {},
            'behavior_similarity': {},
            'final_verdict': {}
        }
    
    def _load_model(self, config: BotConfiguration) -> Dict:
        """
        Loads the model by reconstructing the Python class and loading weights.
        This bypasses the 'RecursiveScriptModule has no forward' crash by using real objects.
        """
        model_dict = {}
        model_path = Path(config.model_path)
        
        # 1. Attempt to find and load the Python Policy Class
        try:
            PolicyClass = config.get_discrete_policy_class()
        except Exception as e:
            Atomic.log(f"could not import policy class: {e}", "WARN")
            return self._load_model_jit_fallback(config)

        if PolicyClass is None:
            return self._load_model_jit_fallback(config)

        Atomic.log(f"reconstructing python model: {PolicyClass.__name__}", "INFO")

        # 2. Instantiate the fresh model
        try:
            full_model = PolicyClass(
                input_shape=config.obs_size,
                n_actions=config.action_size,
                shared_layer_sizes=config.shared_head_layers,
                policy_layer_sizes=config.policy_layers,
                device=self.device
            ).to(self.device)
        except Exception as e:
            Atomic.log(f"model instantiation failed: {e}", "WARN")
            return self._load_model_jit_fallback(config)

        # 3. Load weights from JIT files into the Python object
        try:
            # Handle .lt vs .LT extension hell
            shared_path = model_path / "SHARED_HEAD.lt"
            if not shared_path.exists(): shared_path = model_path / "SHARED_HEAD.LT"
            
            policy_path = model_path / "POLICY.lt"
            if not policy_path.exists(): policy_path = model_path / "POLICY.LT"

            if shared_path.exists():
                # Atomic.log(f"loading SHARED_HEAD from {shared_path.name}", "INFO")
                # Load JIT container just to steal state_dict
                jit_shared = torch.jit.load(str(shared_path), map_location=self.device)
                full_model.shared_head.load_state_dict(jit_shared.state_dict())
                model_dict['shared_head'] = full_model.shared_head
            
            if policy_path.exists():
                # Atomic.log(f"loading POLICY from {policy_path.name}", "INFO")
                jit_policy = torch.jit.load(str(policy_path), map_location=self.device)
                full_model.policy.load_state_dict(jit_policy.state_dict())
                model_dict['policy'] = full_model.policy

            # prevent garbage collection from eating our model
            model_dict['full_instance'] = full_model
            
        except Exception as e:
            Atomic.log(f"transplant failed: {e}", "WARN")
            return self._load_model_jit_fallback(config)

        # Set to eval mode because we aren't training
        full_model.eval()
        return model_dict

    def _load_model_jit_fallback(self, config: BotConfiguration) -> Dict:
        """The old way. It works if you don't look at it too hard."""
        Atomic.log("using JIT fallback loader", "INFO")
        model_dict = {}
        model_path = Path(config.model_path)
        
        shared_path = model_path / "SHARED_HEAD.lt"
        if not shared_path.exists(): shared_path = model_path / "SHARED_HEAD.LT"
        
        if shared_path.exists():
            model_dict['shared_head'] = torch.jit.load(str(shared_path), map_location=self.device)
        
        policy_path = model_path / "POLICY.lt"
        if not policy_path.exists(): policy_path = model_path / "POLICY.LT"
        
        if policy_path.exists():
            model_dict['policy'] = torch.jit.load(str(policy_path), map_location=self.device)
            
        for key in model_dict:
            model_dict[key].eval()
            
        return model_dict
    
    def _get_model_output(self, model_dict: Dict, input_tensor: torch.Tensor) -> torch.Tensor:
        """Get model output"""
        with torch.no_grad():
            x = input_tensor
            if 'shared_head' in model_dict:
                x = model_dict['shared_head'](x)
            if 'policy' in model_dict:
                x = model_dict['policy'](x)
            return x
    
    def _get_all_parameters(self, model_dict: Dict) -> List[torch.Tensor]:
        """Extract all parameters"""
        params = []
        for model_key in ['shared_head', 'policy']:
            if model_key in model_dict:
                for param in model_dict[model_key].parameters():
                    params.append(param.detach().cpu())
        return params
    
    def analyze_weight_similarity(self) -> Dict:
        """
        Analyze weight similarity using Matrix Hunting.
        Instead of assuming layers line up 1:1, we brute force compare every layer.
        """
        Atomic.section("WEIGHT MATRIX HUNT", 1)
        
        # Helper to get named parameters so we know what we found
        def get_named_params(model_dict):
            params = []
            for key in ['shared_head', 'policy']:
                if key in model_dict:
                    for name, param in model_dict[key].named_parameters():
                        # skip 1D tensors (biases, layernorms) usually false positives
                        if len(param.shape) > 1:
                            params.append((f"{key}.{name}", param.detach().cpu()))
            return params

        params1 = get_named_params(self.original_model)
        params2 = get_named_params(self.suspicious_model)
        
        Atomic.log(f"Comparing {len(params1)} original vs {len(params2)} suspect matrices", "INFO")
        
        matches_found = []
        max_similarity = 0.0
        
        # O(N*M) brute force comparison. 
        for name1, p1 in params1:
            p1_flat = p1.flatten().numpy()
            norm1 = np.linalg.norm(p1_flat)
            if norm1 == 0: continue
                
            for name2, p2 in params2:
                # optimization: only compare if shapes match (or are transposed)
                if p1.shape != p2.shape:
                    if p1.shape == p2.T.shape:
                        # check transpose case just to be paranoid
                        p2_check = p2.T.flatten().numpy()
                    else:
                        continue
                else:
                    p2_check = p2.flatten().numpy()

                norm2 = np.linalg.norm(p2_check)
                if norm2 == 0: continue

                # Cosine similarity
                cos_sim = np.dot(p1_flat, p2_check) / (norm1 * norm2)
                
                # We only care about EXACT copies or very close fine-tunes
                if cos_sim > 0.95:
                    matches_found.append({
                        'original': name1,
                        'suspect': name2,
                        'similarity': float(cos_sim)
                    })
                
                max_similarity = max(max_similarity, cos_sim)

        # Analysis logic
        num_matches = len(matches_found)
        total_layers_sus = len(params2)
        match_percentage = (num_matches / total_layers_sus * 100) if total_layers_sus > 0 else 0
        
        self.results['weight_similarity'] = {
            'avg_cosine_similarity': float(max_similarity), 
            'max_single_layer_similarity': float(max_similarity),
            'stolen_layers_count': num_matches,
            'match_details': matches_found
        }
        
        if num_matches > 0:
            Atomic.log(f"FOUND {num_matches} IDENTICAL MATRICES!", "CRIT")
            if match_percentage > 50:
                verdict = "[CRIT] MAJORITY OF BRAIN STOLEN"
                c = Atomic.RED
            else:
                verdict = "[WARN] SURGICAL TRANSPLANT"
                c = Atomic.YELLOW
        else:
            if max_similarity > 0.8:
                verdict = "[SUS] SUSPICIOUS ARCHITECTURE"
                c = Atomic.YELLOW
            else:
                verdict = "[SAFE] NO DIRECT MATCHES"
                c = Atomic.GREEN
        
        Atomic.kv("Max Similarity", f"{max_similarity:.4f}", Atomic.WHITE)
        Atomic.kv("Stolen Layers", f"{num_matches}/{total_layers_sus}", Atomic.RED if num_matches > 0 else Atomic.GREEN)
        Atomic.kv("Verdict", verdict, Atomic.WHITE)
        
        self.results['weight_similarity']['verdict'] = verdict
        return self.results['weight_similarity']
    
    def analyze_activation_similarity(self, num_samples: int = 1000) -> Dict:
        """Analyze activation patterns"""
        Atomic.section("ACTIVATION PATTERNS", 2)
        
        obs_size = min(self.original_cfg.obs_size, self.suspicious_cfg.obs_size)
        random_inputs = torch.randn(num_samples, obs_size).to(self.device)
        
        # Pad/crop to match each model's expected input
        if self.original_cfg.obs_size > obs_size:
            inputs1 = F.pad(random_inputs, (0, self.original_cfg.obs_size - obs_size))
        else:
            inputs1 = random_inputs[:, :self.original_cfg.obs_size]
        
        if self.suspicious_cfg.obs_size > obs_size:
            inputs2 = F.pad(random_inputs, (0, self.suspicious_cfg.obs_size - obs_size))
        else:
            inputs2 = random_inputs[:, :self.suspicious_cfg.obs_size]
        
        # Get outputs
        with torch.no_grad():
            out1 = self._get_model_output(self.original_model, inputs1)
            out2 = self._get_model_output(self.suspicious_model, inputs2)
        
        # Compare
        correlations = []
        cosine_sims = []
        
        for i in range(min(100, num_samples)):
            a1 = out1[i].flatten().cpu().numpy()
            a2 = out2[i].flatten().cpu().numpy()
            
            min_size = min(len(a1), len(a2))
            a1 = a1[:min_size]
            a2 = a2[:min_size]
            
            if len(a1) > 1:
                corr, _ = spearmanr(a1, a2)
                if not np.isnan(corr):
                    correlations.append(corr)
                
                cos_sim = 1 - cosine(a1, a2)
                cosine_sims.append(cos_sim)
        
        avg_corr = np.mean(correlations) if correlations else 0
        avg_cos = np.mean(cosine_sims) if cosine_sims else 0
        
        self.results['activation_similarity'] = {
            'avg_spearman_correlation': float(avg_corr),
            'avg_cosine_similarity': float(avg_cos),
            'num_samples': num_samples
        }
        
        if avg_corr > 0.7:
            verdict = "[HIGH] STRONG CORRELATION"
            c = Atomic.RED
        elif avg_corr > 0.4:
            verdict = "[MED] MODERATE CORRELATION"
            c = Atomic.YELLOW
        else:
            verdict = "[LOW] WEAK CORRELATION"
            c = Atomic.GREEN
        
        Atomic.kv("Spearman Corr", f"{avg_corr:.4f}", c)
        Atomic.kv("Cosine Sim", f"{avg_cos:.4f}", c)
        Atomic.kv("Verdict", verdict, c)
        
        self.results['activation_similarity']['verdict'] = verdict
        return self.results['activation_similarity']
    
    def analyze_gradient_similarity(self, num_samples: int = 50) -> Dict:
        """
        Analyze gradient patterns. 
        Rewritten to manually forward pass because helpers with @no_grad are the devil.
        """
        Atomic.section("GRADIENT ANALYSIS", 3)
        
        obs_size = min(self.original_cfg.obs_size, self.suspicious_cfg.obs_size)
        random_inputs = torch.randn(num_samples, obs_size).to(self.device)
        
        gradient_sims = []
        
        # enable gradients for everyone. oprah style.
        # we iterate through the dictionaries we built in _load_model
        for model_dict in [self.original_model, self.suspicious_model]:
            for key, val in model_dict.items():
                if isinstance(val, torch.nn.Module):
                    for param in val.parameters():
                        param.requires_grad = True
                    val.eval() # keep eval for batchnorm consistency, but grad is on
        
        try:
            for i in range(min(10, num_samples)):
                # 1. prepare inputs manually
                # crop/pad inputs to match weird observation sizes
                if self.original_cfg.obs_size > obs_size:
                    inp1 = F.pad(random_inputs[i:i+1], (0, self.original_cfg.obs_size - obs_size))
                else:
                    inp1 = random_inputs[i:i+1, :self.original_cfg.obs_size]
                
                if self.suspicious_cfg.obs_size > obs_size:
                    inp2 = F.pad(random_inputs[i:i+1], (0, self.suspicious_cfg.obs_size - obs_size))
                else:
                    inp2 = random_inputs[i:i+1, :self.suspicious_cfg.obs_size]
                
                # 2. manual forward pass (CRITICAL: DO NOT USE _get_model_output HERE)
                
                # model 1 forward
                x1 = inp1
                if 'shared_head' in self.original_model:
                    x1 = self.original_model['shared_head'](x1)
                if 'policy' in self.original_model:
                    x1 = self.original_model['policy'](x1)
                
                # model 2 forward
                x2 = inp2
                if 'shared_head' in self.suspicious_model:
                    x2 = self.suspicious_model['shared_head'](x2)
                if 'policy' in self.suspicious_model:
                    x2 = self.suspicious_model['policy'](x2)

                # 3. backward pass
                # we sum the output to get a scalar, then derive
                loss1 = x1.sum()
                loss2 = x2.sum()
                
                # retain_graph=False because we clear anyway, but explicit is nice
                loss1.backward()
                loss2.backward()
                
                # 4. harvest organs (gradients)
                grads1 = []
                grads2 = []
                
                # helper to extract flat grads from a model dict
                def get_flat_grads(m_dict):
                    g = []
                    # strictly ordered keys for consistency
                    for k in ['shared_head', 'policy']:
                        if k in m_dict:
                            for p in m_dict[k].parameters():
                                if p.grad is not None:
                                    g.append(p.grad.flatten().detach().cpu().numpy())
                    return np.concatenate(g) if g else np.array([])

                g1_flat = get_flat_grads(self.original_model)
                g2_flat = get_flat_grads(self.suspicious_model)
                
                if len(g1_flat) > 0 and len(g2_flat) > 0:
                    # align sizes if different layers exist (best effort)
                    min_len = min(len(g1_flat), len(g2_flat))
                    if min_len > 10: # spearman needs some data points
                        corr, _ = spearmanr(g1_flat[:min_len], g2_flat[:min_len])
                        if not np.isnan(corr):
                            gradient_sims.append(corr)
                
                # 5. cleanup on aisle 5
                for m_dict in [self.original_model, self.suspicious_model]:
                    for k, v in m_dict.items():
                        if isinstance(v, torch.nn.Module):
                            v.zero_grad()
        
        except Exception as e:
            Atomic.log(f"gradient calculation exploded: {e}", "WARN")
            self.results['gradient_similarity']['error'] = str(e)
            return self.results['gradient_similarity']
        
        avg_grad_sim = np.mean(gradient_sims) if gradient_sims else 0
        
        self.results['gradient_similarity'] = {
            'avg_gradient_correlation': float(avg_grad_sim),
            'num_samples': len(gradient_sims)
        }
        
        # verdict logic
        if avg_grad_sim > 0.8: verdict = "[HIGH] VERY HIGH CORRELATION"
        elif avg_grad_sim > 0.6: verdict = "[MED] HIGH CORRELATION"
        elif avg_grad_sim > 0.3: verdict = "[LOW] MODERATE CORRELATION"
        else: verdict = "[SAFE] LOW CORRELATION"
        
        c = Atomic.RED if "VERY" in verdict else (Atomic.YELLOW if "MODERATE" in verdict else Atomic.GREEN)
        Atomic.kv("Gradient Corr", f"{avg_grad_sim:.4f}", c)
        Atomic.kv("Verdict", verdict, c)
        
        self.results['gradient_similarity']['verdict'] = verdict
        return self.results['gradient_similarity']
    
    def analyze_distribution_similarity(self, num_samples: int = 1000) -> Dict:
        """Analyze output distributions"""
        Atomic.section("DISTRIBUTION (KL DIV)", 4)
        
        obs_size = min(self.original_cfg.obs_size, self.suspicious_cfg.obs_size)
        random_inputs = torch.randn(num_samples, obs_size).to(self.device)
        
        kl_divs = []
        js_divs = []
        
        with torch.no_grad():
            for i in range(min(100, num_samples)):
                # Prepare inputs
                if self.original_cfg.obs_size > obs_size:
                    inp1 = F.pad(random_inputs[i:i+1], (0, self.original_cfg.obs_size - obs_size))
                else:
                    inp1 = random_inputs[i:i+1, :self.original_cfg.obs_size]
                
                if self.suspicious_cfg.obs_size > obs_size:
                    inp2 = F.pad(random_inputs[i:i+1], (0, self.suspicious_cfg.obs_size - obs_size))
                else:
                    inp2 = random_inputs[i:i+1, :self.suspicious_cfg.obs_size]
                
                # Get outputs
                out1 = self._get_model_output(self.original_model, inp1)
                out2 = self._get_model_output(self.suspicious_model, inp2)
                
                # Convert to probabilities
                prob1 = F.softmax(out1, dim=-1)
                prob2 = F.softmax(out2, dim=-1)
                
                # Align sizes
                min_actions = min(prob1.shape[-1], prob2.shape[-1])
                prob1 = prob1[..., :min_actions]
                prob2 = prob2[..., :min_actions]
                
                # KL Divergence
                kl_div = F.kl_div(prob2.log(), prob1, reduction='batchmean')
                kl_divs.append(kl_div.item())
                
                # JS Divergence
                m = 0.5 * (prob1 + prob2)
                js_div = 0.5 * F.kl_div(m.log(), prob1, reduction='batchmean') + \
                         0.5 * F.kl_div(m.log(), prob2, reduction='batchmean')
                js_divs.append(js_div.item())
        
        avg_kl = np.mean(kl_divs) if kl_divs else float('inf')
        avg_js = np.mean(js_divs) if js_divs else float('inf')
        
        self.results['distribution_similarity'] = {
            'avg_kl_divergence': float(avg_kl),
            'avg_js_divergence': float(avg_js),
            'num_samples': len(kl_divs)
        }
        
        if avg_kl < 0.1:
            verdict = "[HIGH] EXTREME SIMILARITY"
        elif avg_kl < 0.5:
            verdict = "[MED] HIGH SIMILARITY"
        elif avg_kl < 1.5:
            verdict = "[LOW] MODERATE SIMILARITY"
        else:
            verdict = "[SAFE] LOW SIMILARITY"
        
        c = Atomic.RED if "EXTREME" in verdict else (Atomic.YELLOW if "HIGH" in verdict else Atomic.GREEN)
        Atomic.kv("KL Divergence", f"{avg_kl:.4f}", c)
        Atomic.kv("JS Divergence", f"{avg_js:.4f}", c)
        Atomic.kv("Verdict", verdict, c)
        
        self.results['distribution_similarity']['verdict'] = verdict
        return self.results['distribution_similarity']
    
    def analyze_cka_similarity(self, num_samples: int = 500) -> Dict:
        """CKA analysis"""
        Atomic.section("CKA (SOUL CHECK)", 5)
        
        obs_size = min(self.original_cfg.obs_size, self.suspicious_cfg.obs_size)
        random_inputs = torch.randn(num_samples, obs_size).to(self.device)
        
        with torch.no_grad():
            # Model 1
            if self.original_cfg.obs_size > obs_size:
                inp1 = F.pad(random_inputs, (0, self.original_cfg.obs_size - obs_size))
            else:
                inp1 = random_inputs[:, :self.original_cfg.obs_size]
            
            repr1 = inp1
            if 'shared_head' in self.original_model:
                repr1 = self.original_model['shared_head'](repr1)
            
            # Model 2
            if self.suspicious_cfg.obs_size > obs_size:
                inp2 = F.pad(random_inputs, (0, self.suspicious_cfg.obs_size - obs_size))
            else:
                inp2 = random_inputs[:, :self.suspicious_cfg.obs_size]
            
            repr2 = inp2
            if 'shared_head' in self.suspicious_model:
                repr2 = self.suspicious_model['shared_head'](repr2)
        
        # Convert to numpy
        X = repr1.cpu().numpy()
        Y = repr2.cpu().numpy()
        
        # Center
        X = X - X.mean(axis=0, keepdims=True)
        Y = Y - Y.mean(axis=0, keepdims=True)
        
        # Compute gram matrices
        K = X @ X.T
        L = Y @ Y.T
        
        # CKA
        hsic = np.trace(K @ L)
        normalization = np.sqrt(np.trace(K @ K) * np.trace(L @ L))
        cka = hsic / (normalization + 1e-10)
        
        self.results['cka_similarity'] = {
            'cka_score': float(cka),
            'num_samples': num_samples
        }
        
        if cka > 0.9:
            verdict = "[HIGH] IDENTICAL FEATURE EXTRACTION"
        elif cka > 0.7:
            verdict = "[MED] SIMILAR FEATURE EXTRACTION"
        elif cka > 0.5:
            verdict = "[LOW] VAGUE RESEMBLANCE"
        else:
            verdict = "[SAFE] DISTINCT"
        
        c = Atomic.RED if cka > 0.9 else (Atomic.YELLOW if cka > 0.5 else Atomic.GREEN)
        Atomic.kv("CKA Score", f"{cka:.4f}", c)
        Atomic.kv("Verdict", verdict, c)
        
        self.results['cka_similarity']['verdict'] = verdict
        return self.results['cka_similarity']
    
    def analyze_behavior_similarity(self, num_samples: int = 50000) -> Dict:
        """
        Analyze behavioral similarity.
        BRUTE FORCE MODE: checking 50k+ samples because paranoia is a virtue.
        Vectorized to prevent your CPU from melting.
        """
        Atomic.section("BEHAVIOR (ACTIONS)", 6)
        
        if self.original_act is None or self.suspicious_act is None:
            Atomic.log("missing action parsers, skipping behavior", "WARN")
            self.results['behavior_similarity']['status'] = 'unavailable'
            return self.results['behavior_similarity']
        
        # use the smallest obs size to generate valid noise
        base_obs_size = min(self.original_cfg.obs_size, self.suspicious_cfg.obs_size)
        
        # stats trackers
        total_agreement_count = 0
        total_samples_processed = 0
        prob_correlations = []
        
        # batch size for processing (chunks of 2048 are usually safe for cpu inference)
        batch_size = 2048
        
        # helper to batch pad/crop numpy arrays
        def get_batch_obs(data, target_size):
            # data shape: (Batch, BaseSize)
            if target_size == data.shape[1]:
                return data
            if target_size > data.shape[1]:
                # pad columns with zeros
                padding = np.zeros((data.shape[0], target_size - data.shape[1]), dtype=np.float32)
                return np.hstack([data, padding])
            else:
                # crop columns
                return data[:, :target_size]
        
        with torch.no_grad():
            for i in range(0, num_samples, batch_size):
                current_batch_size = min(batch_size, num_samples - i)
                
                # generate random noise batch (clamped to look vaguely like game data)
                random_obs = np.random.randn(current_batch_size, base_obs_size).astype(np.float32)
                random_obs = np.clip(random_obs, -3.0, 3.0)
                
                # prepare inputs for both bots
                obs1 = get_batch_obs(random_obs, self.original_cfg.obs_size)
                obs2 = get_batch_obs(random_obs, self.suspicious_cfg.obs_size)
                
                t1 = torch.from_numpy(obs1).to(self.device)
                t2 = torch.from_numpy(obs2).to(self.device)
                
                # inference
                out1 = self._get_model_output(self.original_model, t1)
                out2 = self._get_model_output(self.suspicious_model, t2)
                
                # get probs and raw indices
                prob1 = F.softmax(out1, dim=-1).cpu().numpy()
                prob2 = F.softmax(out2, dim=-1).cpu().numpy()
                
                idx1 = np.argmax(prob1, axis=1)
                idx2 = np.argmax(prob2, axis=1)
                
                try:
                    # vectorized action parsing (assuming parse_actions handles lists/arrays)
                    # most lookup parsers just do table[indices], which supports arrays natively
                    act1_game = self.original_act.parse_actions(idx1)
                    act2_game = self.suspicious_act.parse_actions(idx2)
                    
                    # check for agreement
                    # if the output is (Batch, 8), we check if all 8 params match per row
                    if act1_game.shape == act2_game.shape:
                        # returns boolean array of shape (Batch,)
                        # atol=0.1 allows for tiny float errors
                        matches = np.all(np.isclose(act1_game, act2_game, atol=0.1), axis=1)
                        total_agreement_count += np.sum(matches)
                    else:
                        # shape mismatch (different action sets?) -> fallback to raw index match
                        # strictly speaking this is unfair if shapes differ, but it's a fallback
                        matches = (idx1 == idx2)
                        total_agreement_count += np.sum(matches)
                        
                except Exception as e:
                    # if batch parsing fails, fallback to simple index comparison
                    # (some parsers might be bad and not support vectorization)
                    matches = (idx1 == idx2)
                    total_agreement_count += np.sum(matches)

                # calculate correlation only on the first batch to save time
                # (doing spearman on 50k items is slow and unnecessary)
                if i == 0:
                    # calc mean correlation of the probability distributions
                    # we do this for a subset (first 100) of the first batch
                    subset = min(100, current_batch_size)
                    min_act_cols = min(prob1.shape[1], prob2.shape[1])
                    for k in range(subset):
                        corr, _ = spearmanr(prob1[k, :min_act_cols], prob2[k, :min_act_cols])
                        if not np.isnan(corr):
                            prob_correlations.append(corr)

                total_samples_processed += current_batch_size
                
                # progress indicator for impatient devs
                if (i // batch_size) % 5 == 0 and i > 0:
                    pass

        print("") # newline after progress dots
        
        agreement_rate = total_agreement_count / total_samples_processed
        avg_correlation = np.mean(prob_correlations) if prob_correlations else 0
        
        self.results['behavior_similarity'] = {
            'action_agreement_rate': float(agreement_rate),
            'avg_prob_correlation': float(avg_correlation),
            'samples_checked': total_samples_processed
        }
        
        if agreement_rate > 0.9: verdict = "[HIGH] IDENTICAL ACTIONS"
        elif agreement_rate > 0.7: verdict = "[MED] HIGH AGREEMENT"
        elif agreement_rate > 0.4: verdict = "[LOW] MODERATE AGREEMENT"
        else: verdict = "[SAFE] RANDOM/DISTINCT"
        
        c = Atomic.RED if "IDENTICAL" in verdict else (Atomic.YELLOW if "HIGH" in verdict else Atomic.GREEN)
        Atomic.kv("Samples Checked", f"{total_samples_processed}", Atomic.CYAN)
        Atomic.kv("Action Agreement", f"{agreement_rate:.4f}", c)
        Atomic.kv("Prob Correlation", f"{avg_correlation:.4f}", c)
        Atomic.kv("Verdict", verdict, c)
        
        self.results['behavior_similarity']['verdict'] = verdict
        return self.results['behavior_similarity']
    
    def compute_final_verdict(self) -> Dict:
        """
        Compute final verdict.
        updated logic: if the fingerprints match, i don't care if the shoes don't fit.
        """
        Atomic.section("FINAL JUDGMENT", 7)
        
        score = 0
        max_score = 0
        evidence = []
        smoking_gun = False
        
        # --- PHASE 1: THE "SMOKING GUN" CHECK ---
        weight_sim = self.results.get('weight_similarity', {}).get('avg_cosine_similarity', 0)
        cka_score = self.results.get('cka_similarity', {}).get('cka_score', 0)
        
        if weight_sim > 0.95:
            smoking_gun = True
            evidence.append(f"[!!!] SMOKING GUN: Weights are {weight_sim*100:.1f}% identical.")
        
        if cka_score > 0.98:
            smoking_gun = True
            evidence.append(f"[!!!] SMOKING GUN: Internal Representation (CKA) is {cka_score:.4f}.")

        # --- PHASE 2: POINT SCORING ---
        
        # 1. Weight Similarity (10 points)
        if 'verdict' in self.results['weight_similarity']:
            max_score += 10
            verdict = self.results['weight_similarity']['verdict']
            if 'CRIT' in verdict: score += 10
            elif 'WARN' in verdict: score += 6
            elif 'SUS' in verdict: score += 3
            if 'CRIT' in verdict or 'WARN' in verdict:
                evidence.append(f"Weight Similarity: {verdict}")
        
        # 2. CKA Similarity (8 points)
        if 'verdict' in self.results['cka_similarity']:
            max_score += 8
            verdict = self.results['cka_similarity']['verdict']
            if 'HIGH' in verdict: score += 8
            elif 'MED' in verdict: score += 5
            elif 'LOW' in verdict: score += 2
            if 'HIGH' in verdict or 'MED' in verdict:
                evidence.append(f"CKA Similarity: {verdict}")

        # 3. Distribution (5 points)
        if 'verdict' in self.results['distribution_similarity']:
            max_score += 5
            verdict = self.results['distribution_similarity']['verdict']
            if 'HIGH' in verdict: score += 5
            elif 'MED' in verdict: score += 3
            elif 'LOW' in verdict: score += 1
            if 'HIGH' in verdict:
                evidence.append(f"Distribution Similarity: {verdict}")

        # 4. Activation (3 points)
        if 'verdict' in self.results['activation_similarity']:
            max_score += 3
            verdict = self.results['activation_similarity']['verdict']
            if 'HIGH' in verdict: score += 3
            elif 'MED' in verdict: score += 1

        # 5. Behavior & Gradients (2 points each)
        for key, name in [('behavior_similarity', 'Behavior'), ('gradient_similarity', 'Gradient')]:
            if 'verdict' in self.results[key]:
                max_score += 2
                verdict = self.results[key]['verdict']
                if 'HIGH' in verdict: score += 2
                elif 'MED' in verdict: score += 1
        
        # --- PHASE 3: CALCULATE CONFIDENCE ---
        
        if smoking_gun:
            confidence = 99.9
            final_verdict = "CONFIRMED COPY / TRANSFER LEARNING"
            color = Atomic.RED
        else:
            confidence = (score / max_score * 100) if max_score > 0 else 0
            
            if confidence >= 85:
                final_verdict = "HIGHLY LIKELY TRANSFER LEARNING"
                color = Atomic.RED
            elif confidence >= 65:
                final_verdict = "LIKELY TRANSFER LEARNING"
                color = Atomic.YELLOW
            elif confidence >= 40:
                final_verdict = "POSSIBLE TRANSFER LEARNING"
                color = Atomic.YELLOW
            elif confidence >= 20:
                final_verdict = "UNLIKELY"
                color = Atomic.GREEN
            else:
                final_verdict = "NO EVIDENCE"
                color = Atomic.GREEN
        
        self.results['final_verdict'] = {
            'verdict': final_verdict,
            'confidence_percentage': float(confidence),
            'score': score,
            'max_score': max_score,
            'evidence': evidence,
            'is_smoking_gun': smoking_gun
        }
        
        print(f"\n{Atomic.PURPLE}========================================================{Atomic.RESET}")
        print(f"  STATUS:     {color}{Atomic.BOLD}{final_verdict}{Atomic.RESET}")
        print(f"  CONFIDENCE: {color}{confidence:.1f}%{Atomic.RESET}")
        print(f"{Atomic.PURPLE}========================================================{Atomic.RESET}")
            
        print(f"\n{Atomic.GREY}EVIDENCE LOCKER:{Atomic.RESET}")
        for e in evidence:
            print(f"  - {e}")
        if not evidence:
            print("  - No significant evidence detected")
        print("")
        
        return self.results['final_verdict']
    
    def run_full_analysis(self) -> Dict:
        """Run complete analysis"""
        Atomic.banner()
        Atomic.log(f"Original: {self.original_cfg.bot_name}", "INFO")
        Atomic.log(f"Suspect:  {self.suspicious_cfg.bot_name}", "INFO")
        
        try:
            self.analyze_weight_similarity()
        except Exception as e:
            Atomic.log(f"Weight analysis failed: {e}", "CRIT")
        
        try:
            self.analyze_activation_similarity()
        except Exception as e:
            Atomic.log(f"Activation analysis failed: {e}", "CRIT")
        
        try:
            self.analyze_gradient_similarity()
        except Exception as e:
            Atomic.log(f"Gradient analysis failed: {e}", "CRIT")
        
        try:
            self.analyze_distribution_similarity()
        except Exception as e:
            Atomic.log(f"Distribution analysis failed: {e}", "CRIT")
        
        try:
            self.analyze_cka_similarity()
        except Exception as e:
            Atomic.log(f"CKA analysis failed: {e}", "CRIT")
        
        try:
            self.analyze_behavior_similarity()
        except Exception as e:
            Atomic.log(f"Behavior analysis failed: {e}", "CRIT")
        
        self.compute_final_verdict()
        
        return self.results
    
    def save_results(self, output_path: str):
        """Save results to JSON"""
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        Atomic.log(f"results saved to: {output_path}", "INFO")


def main():
    parser = argparse.ArgumentParser(
        description='Transfer Learning Detection for GigaLearnCPP Bots'
    )
    parser.add_argument('original_json', help='Path to original bot JSON config')
    parser.add_argument('sus_json', help='Path to suspicious bot JSON config')
    parser.add_argument('--output', '-o', default='tl_detection_results.json',
                        help='Output JSON file path')
    parser.add_argument('--device', '-d', default='cpu', choices=['cpu', 'cuda'],
                        help='Device to use for analysis')
    
    args = parser.parse_args()
    
    # Load configurations
    # Atomic.log("Parsing configs...", "INFO")
    try:
        original_cfg = BotConfiguration(args.original_json)
        suspicious_cfg = BotConfiguration(args.sus_json)
    except Exception as e:
        Atomic.log(f"Config load failed: {e}", "CRIT")
        sys.exit(1)
    
    # Create detector
    detector = TransferLearningDetector(
        original_cfg,
        suspicious_cfg,
        device=args.device
    )
    
    # Run analysis
    results = detector.run_full_analysis()
    
    # Save results
    detector.save_results(args.output)
    
    Atomic.log("job done. closing connection.", "GOD")


if __name__ == "__main__":
    main()