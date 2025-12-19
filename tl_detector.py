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
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.stats import spearmanr, pearsonr, ks_2samp
from scipy.spatial.distance import cosine, euclidean
from scipy.linalg import svd
import warnings
warnings.filterwarnings('ignore')


def generate_kickoff_states(num_states=100):
    """
    Generate realistic kickoff observation states based on GigaLearnCPP/RLGymCPP
    
    AdvancedObs structure (109 elements for 1v1):
    - Ball: pos(3) + vel(3) + angVel(3) = 9
    - PrevAction: 8 elements
    - BoostPads: 34 elements (timers)
    - Self: pos(3) + forward(3) + up(3) + vel(3) + angVel(3) + localAngVel(3) + 
            localBallPos(3) + localBallVel(3) + boost(1) + onGround(1) + hasFlip(1) + isDemoed(1) + hasJumped(1) = 29
    - Opponent: same 29 elements
    Total: 9 + 8 + 34 + 29 + 29 = 109
    """
    states = []
    
    # Kickoff positions (5 positions per team in standard Rocket League)
    kickoff_positions = [
        # Diagonal left
        np.array([-2048, -2560, 17], dtype=np.float32),
        # Diagonal right
        np.array([2048, -2560, 17], dtype=np.float32),
        # Back left
        np.array([-256, -3840, 17], dtype=np.float32),
        # Back right
        np.array([256, -3840, 17], dtype=np.float32),
        # Center
        np.array([0, -4608, 17], dtype=np.float32)
    ]
    
    for _ in range(num_states):
        obs = np.zeros(109, dtype=np.float32)
        
        # Ball at center (normalized)
        obs[0:3] = [0, 0, 93/5000]  # Center of field, on ground
        obs[3:6] = [0, 0, 0]  # No velocity at kickoff
        obs[6:9] = [0, 0, 0]  # No angular velocity
        
        # Previous action (zeros at kickoff)
        obs[9:17] = [0] * 8
        
        # Boost pads (all available at kickoff)
        obs[17:51] = [1.0] * 34
        
        # Self player - random kickoff position (blue team)
        pos_idx = np.random.randint(0, len(kickoff_positions))
        pos = kickoff_positions[pos_idx].copy()
        # Add small noise
        pos += np.random.randn(3) * 50
        
        # Normalize position
        obs[51:54] = pos / 5000
        
        # Forward vector (facing ball)
        forward = np.array([0, 1, 0]) + np.random.randn(3) * 0.05
        forward = forward / (np.linalg.norm(forward) + 1e-8)
        obs[54:57] = forward
        
        # Up vector
        obs[57:60] = [0, 0, 1]
        
        # Velocity (small at kickoff)
        obs[60:63] = np.random.randn(3) * 0.01
        
        # Angular velocity (small)
        obs[63:66] = np.random.randn(3) * 0.01
        obs[66:69] = np.random.randn(3) * 0.01  # Local angular velocity
        
        # Local ball position
        local_ball = np.array([0, 0, 93]) - pos
        obs[69:72] = local_ball / 5000
        
        # Local ball velocity
        obs[72:75] = [0, 0, 0]
        
        # Boost amount (usually 33 at kickoff)
        obs[75] = 0.33 + np.random.rand() * 0.01
        
        # On ground
        obs[76] = 1.0
        
        # Has flip
        obs[77] = 1.0
        
        # Is demoed
        obs[78] = 0.0
        
        # Has jumped
        obs[79] = 0.0
        
        # Opponent (orange team) - opposite side
        opp_pos = -pos.copy()
        opp_pos += np.random.randn(3) * 50
        obs[80:83] = opp_pos / 5000
        
        # Opponent forward (facing ball)
        opp_forward = np.array([0, -1, 0]) + np.random.randn(3) * 0.05
        opp_forward = opp_forward / (np.linalg.norm(opp_forward) + 1e-8)
        obs[83:86] = opp_forward
        
        # Opponent up
        obs[86:89] = [0, 0, 1]
        
        # Opponent velocities (small)
        obs[89:92] = np.random.randn(3) * 0.01
        obs[92:95] = np.random.randn(3) * 0.01
        obs[95:98] = np.random.randn(3) * 0.01
        
        # Opponent local ball
        opp_local_ball = np.array([0, 0, 93]) - opp_pos
        obs[98:101] = opp_local_ball / 5000
        obs[101:104] = [0, 0, 0]
        
        # Opponent boost
        obs[104] = 0.33 + np.random.rand() * 0.01
        obs[105] = 1.0  # On ground
        obs[106] = 1.0  # Has flip
        obs[107] = 0.0  # Not demoed
        obs[108] = 0.0  # Hasn't jumped
        
        states.append(obs)
    
    return np.array(states, dtype=np.float32)


def generate_realistic_game_states(num_states=100):
    """Generate realistic mid-game states"""
    states = []
    
    for _ in range(num_states):
        obs = np.zeros(109, dtype=np.float32)
        
        # Ball in random position
        obs[0] = np.random.uniform(-4096, 4096) / 5000  # X
        obs[1] = np.random.uniform(-5120, 5120) / 5000  # Y
        obs[2] = np.random.uniform(93, 2000) / 5000     # Z
        
        # Ball velocity
        obs[3:6] = np.random.randn(3) * 0.5
        
        # Ball angular velocity
        obs[6:9] = np.random.randn(3) * 0.3
        
        # Previous action (random)
        obs[9:17] = np.random.uniform(-1, 1, 8)
        
        # Boost pads (random availability)
        obs[17:51] = np.random.rand(34)
        
        # Self player
        obs[51:54] = np.random.uniform(-1, 1, 3)  # Position
        forward = np.random.randn(3)
        forward = forward / (np.linalg.norm(forward) + 1e-8)
        obs[54:57] = forward
        up = np.random.randn(3)
        up = up / (np.linalg.norm(up) + 1e-8)
        obs[57:60] = up
        obs[60:63] = np.random.randn(3) * 0.5  # Velocity
        obs[63:69] = np.random.randn(6) * 0.3  # Angular velocities
        obs[69:75] = np.random.randn(6) * 0.5  # Local ball pos/vel
        obs[75] = np.random.rand()  # Boost
        obs[76] = np.random.choice([0, 1])  # On ground
        obs[77] = np.random.choice([0, 1])  # Has flip
        obs[78] = 0.0  # Is demoed
        obs[79] = np.random.choice([0, 1])  # Has jumped
        
        # Opponent (similar)
        obs[80:109] = obs[51:80].copy()
        obs[80:83] += np.random.randn(3) * 0.2  # Different position
        
        states.append(obs)
    
    return np.array(states, dtype=np.float32)


class AdvancedDetectionMethods:
    """Additional detection methods for the main detector"""
    
    @staticmethod
    def analyze_layer_features(model1_dict, model2_dict, device, num_samples=500):
        """
        Extract and compare features learned by each layer
        Returns correlation of feature representations
        """
        results = {'method': 'layer_features'}
        
        # Generate test inputs
        test_inputs = torch.randn(num_samples, 109).to(device)
        
        layer_correlations = []
        
        with torch.no_grad():
            # Get all intermediate activations
            acts1 = []
            acts2 = []
            
            # Model 1
            x1 = test_inputs
            if 'shared_head' in model1_dict:
                for layer in model1_dict['shared_head']:
                    if isinstance(layer, nn.Linear):
                        x1 = layer(x1)
                        acts1.append(x1.clone())
                    elif isinstance(layer, nn.ReLU):
                        x1 = layer(x1)
            
            # Model 2
            x2 = test_inputs
            if 'shared_head' in model2_dict:
                for layer in model2_dict['shared_head']:
                    if isinstance(layer, nn.Linear):
                        x2 = layer(x2)
                        acts2.append(x2.clone())
                    elif isinstance(layer, nn.ReLU):
                        x2 = layer(x2)
            
            # Compare layer-wise
            for i, (a1, a2) in enumerate(zip(acts1, acts2)):
                # Compute correlation of feature activations
                a1_flat = a1.flatten().cpu().numpy()
                a2_flat = a2.flatten().cpu().numpy()
                
                min_len = min(len(a1_flat), len(a2_flat))
                if min_len > 10:
                    corr, _ = spearmanr(a1_flat[:min_len], a2_flat[:min_len])
                    if not np.isnan(corr):
                        layer_correlations.append({
                            'layer': i,
                            'correlation': float(corr)
                        })
        
        results['layer_correlations'] = layer_correlations
        results['avg_layer_correlation'] = float(np.mean([l['correlation'] for l in layer_correlations])) if layer_correlations else 0
        
        return results
    
    @staticmethod
    def analyze_decision_boundaries(model1_dict, model2_dict, device, obs_size, num_samples=200):
        """
        Compare decision boundaries by testing around decision points
        """
        results = {'method': 'decision_boundaries'}
        
        # Generate base states
        base_states = torch.randn(num_samples, obs_size).to(device)
        
        boundary_agreements = []
        
        with torch.no_grad():
            # Get predictions
            def get_pred(model_dict, x):
                if 'shared_head' in model_dict:
                    x = model_dict['shared_head'](x)
                if 'policy' in model_dict:
                    x = model_dict['policy'](x)
                return torch.argmax(x, dim=-1)
            
            pred1 = get_pred(model1_dict, base_states)
            pred2 = get_pred(model2_dict, base_states)
            
            # Test perturbations around each state
            for i in range(min(50, num_samples)):
                base = base_states[i:i+1]
                
                # Generate perturbations
                perturbations = []
                for _ in range(10):
                    noise = torch.randn_like(base) * 0.1
                    perturbations.append(base + noise)
                
                perturbations = torch.cat(perturbations, dim=0)
                
                # Get predictions on perturbations
                perturbed_pred1 = get_pred(model1_dict, perturbations)
                perturbed_pred2 = get_pred(model2_dict, perturbations)
                
                # Check if decision boundaries are similar
                agreement = (perturbed_pred1 == perturbed_pred2).float().mean().item()
                boundary_agreements.append(agreement)
        
        results['boundary_agreement'] = float(np.mean(boundary_agreements)) if boundary_agreements else 0
        results['boundary_std'] = float(np.std(boundary_agreements)) if boundary_agreements else 0
        
        return results
    
    @staticmethod
    def analyze_neuron_activation_patterns(model1_dict, model2_dict, device, num_samples=500):
        """
        Analyze which neurons activate together (co-activation patterns)
        """
        results = {'method': 'neuron_coactivation'}
        
        test_inputs = torch.randn(num_samples, 109).to(device)
        
        def get_activations(model_dict, x):
            activations = []
            if 'shared_head' in model_dict:
                for layer in model_dict['shared_head']:
                    x = layer(x)
                    if isinstance(layer, nn.ReLU):
                        # Record binary activation (neuron on/off)
                        activations.append((x > 0).float())
            return activations
        
        with torch.no_grad():
            acts1 = get_activations(model1_dict, test_inputs)
            acts2 = get_activations(model2_dict, test_inputs)
            
            pattern_similarities = []
            
            for a1, a2 in zip(acts1, acts2):
                # Compute co-activation matrix
                # (which pairs of neurons activate together)
                coact1 = (a1.T @ a1) / num_samples  # Correlation matrix
                coact2 = (a2.T @ a2) / num_samples
                
                # Flatten and compare
                c1_flat = coact1.flatten().cpu().numpy()
                c2_flat = coact2.flatten().cpu().numpy()
                
                min_len = min(len(c1_flat), len(c2_flat))
                if min_len > 100:
                    corr, _ = spearmanr(c1_flat[:min_len], c2_flat[:min_len])
                    if not np.isnan(corr):
                        pattern_similarities.append(float(corr))
        
        results['coactivation_similarity'] = float(np.mean(pattern_similarities)) if pattern_similarities else 0
        
        return results
    
    @staticmethod
    def analyze_temporal_consistency(model1_dict, model2_dict, device, obs_builder1=None, obs_builder2=None):
        """
        Test how decisions evolve over a sequence of game states
        """
        results = {'method': 'temporal_consistency'}
        
        # Generate a sequence of states (simulating a game progression)
        num_sequences = 20
        sequence_length = 10
        
        sequence_agreements = []
        
        with torch.no_grad():
            for _ in range(num_sequences):
                # Start with kickoff
                kickoff = generate_kickoff_states(1)[0]
                
                # Evolve state
                sequence = [kickoff]
                for _ in range(sequence_length - 1):
                    # Apply random "physics" progression
                    next_state = sequence[-1].copy()
                    # Update ball position
                    next_state[0:3] += next_state[3:6] * 0.1  # pos += vel * dt
                    # Add noise
                    next_state += np.random.randn(109) * 0.05
                    # Clip to valid ranges
                    next_state = np.clip(next_state, -2, 2)
                    sequence.append(next_state)
                
                sequence = torch.from_numpy(np.array(sequence)).float().to(device)
                
                # Get action sequences from both models
                def get_action_sequence(model_dict, states):
                    actions = []
                    for state in states:
                        x = state.unsqueeze(0)
                        if 'shared_head' in model_dict:
                            x = model_dict['shared_head'](x)
                        if 'policy' in model_dict:
                            x = model_dict['policy'](x)
                        action = torch.argmax(x, dim=-1).item()
                        actions.append(action)
                    return actions
                
                seq1 = get_action_sequence(model1_dict, sequence)
                seq2 = get_action_sequence(model2_dict, sequence)
                
                # Compare sequences
                agreement = sum(a1 == a2 for a1, a2 in zip(seq1, seq2)) / sequence_length
                sequence_agreements.append(agreement)
        
        results['temporal_agreement'] = float(np.mean(sequence_agreements)) if sequence_agreements else 0
        results['temporal_std'] = float(np.std(sequence_agreements)) if sequence_agreements else 0
        
        return results
    
    @staticmethod
    def analyze_kickoff_behavior(model1_dict, model2_dict, device, num_kickoffs=100):
        """
        Specifically analyze behavior on kickoff states
        This is realistic and important for RL bots
        """
        results = {'method': 'kickoff_analysis'}
        
        # Generate kickoff states
        kickoff_states = generate_kickoff_states(num_kickoffs)
        kickoff_tensor = torch.from_numpy(kickoff_states).to(device)
        
        with torch.no_grad():
            # Get predictions
            def get_probs(model_dict, x):
                if 'shared_head' in model_dict:
                    x = model_dict['shared_head'](x)
                if 'policy' in model_dict:
                    x = model_dict['policy'](x)
                return F.softmax(x, dim=-1)
            
            probs1 = get_probs(model1_dict, kickoff_tensor)
            probs2 = get_probs(model2_dict, kickoff_tensor)
            
            # Compare action distributions on kickoffs
            actions1 = torch.argmax(probs1, dim=-1).cpu().numpy()
            actions2 = torch.argmax(probs2, dim=-1).cpu().numpy()
            
            # Action agreement
            agreement = np.mean(actions1 == actions2)
            
            # KL divergence on probability distributions
            kl_divs = []
            for p1, p2 in zip(probs1, probs2):
                min_actions = min(len(p1), len(p2))
                p1_trim = p1[:min_actions]
                p2_trim = p2[:min_actions]
                kl = F.kl_div(p2_trim.log(), p1_trim, reduction='sum').item()
                kl_divs.append(kl)
            
            avg_kl = np.mean(kl_divs)
            
            # Analyze action diversity
            unique_actions1 = len(np.unique(actions1))
            unique_actions2 = len(np.unique(actions2))
            
        results['kickoff_action_agreement'] = float(agreement)
        results['kickoff_avg_kl_divergence'] = float(avg_kl)
        results['kickoff_action_diversity_model1'] = int(unique_actions1)
        results['kickoff_action_diversity_model2'] = int(unique_actions2)
        
        return results
    
    @staticmethod
    def analyze_robustness(model1_dict, model2_dict, device, obs_size, num_samples=100):
        """
        Test adversarial robustness - similar robustness suggests similar decision boundaries
        """
        results = {'method': 'robustness_testing'}
        
        base_states = torch.randn(num_samples, obs_size).to(device)
        
        robustness_correlations = []
        
        with torch.no_grad():
            def get_confidence(model_dict, x):
                if 'shared_head' in model_dict:
                    x = model_dict['shared_head'](x)
                if 'policy' in model_dict:
                    x = model_dict['policy'](x)
                probs = F.softmax(x, dim=-1)
                return torch.max(probs, dim=-1)[0]
            
            # Test with increasing noise levels
            noise_levels = [0.01, 0.05, 0.1, 0.2, 0.5]
            
            for noise_std in noise_levels:
                noise = torch.randn_like(base_states) * noise_std
                noisy_states = base_states + noise
                
                # Get confidence on noisy inputs
                conf1 = get_confidence(model1_dict, noisy_states).cpu().numpy()
                conf2 = get_confidence(model2_dict, noisy_states).cpu().numpy()
                
                # Correlate robustness
                corr, _ = spearmanr(conf1, conf2)
                if not np.isnan(corr):
                    robustness_correlations.append({
                        'noise_level': noise_std,
                        'confidence_correlation': float(corr)
                    })
        
        results['robustness_correlations'] = robustness_correlations
        results['avg_robustness_correlation'] = float(np.mean([r['confidence_correlation'] for r in robustness_correlations])) if robustness_correlations else 0
        
        return results
    
    @staticmethod
    def detect_transfer_learning_signature(model1_dict, model2_dict):
        """
        Look for specific signatures of transfer learning:
        - Early layers very similar, later layers different
        - Similar low-level features, different high-level features
        """
        results = {'method': 'transfer_learning_signature'}
        
        def get_layer_weights(model_dict):
            weights = []
            for key in ['shared_head', 'policy']:
                if key in model_dict:
                    for name, param in model_dict[key].named_parameters():
                        if 'weight' in name:
                            weights.append(param.detach().cpu().numpy())
            return weights
        
        weights1 = get_layer_weights(model1_dict)
        weights2 = get_layer_weights(model2_dict)
        
        if len(weights1) != len(weights2):
            results['signature_detected'] = False
            results['reason'] = 'different_architectures'
            return results
        
        # Compute similarity for each layer
        layer_similarities = []
        for w1, w2 in zip(weights1, weights2):
            if w1.shape == w2.shape:
                # Flatten and compare
                w1_flat = w1.flatten()
                w2_flat = w2.flatten()
                sim = 1 - cosine(w1_flat, w2_flat)
                layer_similarities.append(sim)
            else:
                layer_similarities.append(0)
        
        results['layer_similarities'] = [float(s) for s in layer_similarities]
        
        # Check for transfer learning pattern
        if len(layer_similarities) >= 3:
            early_avg = np.mean(layer_similarities[:len(layer_similarities)//2])
            late_avg = np.mean(layer_similarities[len(layer_similarities)//2:])
            
            # Transfer learning signature: early layers similar, later different
            if early_avg > 0.8 and late_avg < 0.6:
                results['signature_detected'] = True
                results['pattern'] = 'early_frozen_late_trained'
                results['early_similarity'] = float(early_avg)
                results['late_similarity'] = float(late_avg)
            elif early_avg > 0.9 and late_avg > 0.9:
                results['signature_detected'] = True
                results['pattern'] = 'full_model_copy'
                results['early_similarity'] = float(early_avg)
                results['late_similarity'] = float(late_avg)
            else:
                results['signature_detected'] = False
                results['early_similarity'] = float(early_avg)
                results['late_similarity'] = float(late_avg)
        else:
            results['signature_detected'] = False
            results['reason'] = 'insufficient_layers'
        
        return results


# Export for use in main detector
__all__ = [
    'AdvancedDetectionMethods',
    'generate_kickoff_states',
    'generate_realistic_game_states'
]

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
        idx_str = f"[{index}/11] " if index else ""
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
        Atomic.section("FINAL JUDGMENT", 11)
        
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
        
        # ==================== NEW ADVANCED METHODS SCORING ====================
        # 6. Kickoff Analysis (4 points)
        if 'kickoff_analysis' in self.results and 'verdict' in self.results['kickoff_analysis']:
            max_score += 4
            verdict = self.results['kickoff_analysis']['verdict']
            if 'CRIT' in verdict: 
                score += 4
                evidence.append(f"Kickoff Behavior: {verdict}")
            elif 'WARN' in verdict: 
                score += 2
                evidence.append(f"Kickoff Behavior: {verdict}")
        
        # 7. Transfer Learning Signature (5 points - CRITICAL)
        if 'transfer_signature' in self.results and 'verdict' in self.results.get('transfer_signature', {}):
            max_score += 5
            verdict = self.results['transfer_signature'].get('verdict', '')
            if 'CRIT' in verdict:
                score += 5
                evidence.append(f"TL Signature: {verdict}")
        
        # 8. Eigenvalue Spectrum (3 points)
        if 'eigenvalue_spectrum' in self.results and 'verdict' in self.results['eigenvalue_spectrum']:
            max_score += 3
            verdict = self.results['eigenvalue_spectrum']['verdict']
            if 'CRIT' in verdict:
                score += 3
                evidence.append(f"Eigenvalue Spectrum: {verdict}")
            elif 'WARN' in verdict:
                score += 2
        
        # 9. Decision Boundaries (3 points)
        if 'decision_boundaries' in self.results and 'verdict' in self.results['decision_boundaries']:
            max_score += 3
            verdict = self.results['decision_boundaries']['verdict']
            if 'CRIT' in verdict:
                score += 3
                evidence.append(f"Decision Boundaries: {verdict}")
            elif 'WARN' in verdict:
                score += 2
        
        # 10. Temporal Consistency (2 points)
        if 'temporal_consistency' in self.results and 'verdict' in self.results['temporal_consistency']:
            max_score += 2
            verdict = self.results['temporal_consistency']['verdict']
            if 'CRIT' in verdict:
                score += 2
                evidence.append(f"Temporal Behavior: {verdict}")
            elif 'WARN' in verdict:
                score += 1
        # ==================== END NEW METHODS SCORING ====================
        
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
    

    # ==================== ADVANCED DETECTION METHODS ====================
    # Added by integrate_advanced_methods.py
    
    def analyze_kickoff_behavior(self, num_kickoffs: int = 100) -> Dict:
        """
        [NEW METHOD 1/8] Analyze behavior specifically on kickoff states
        Uses realistic Rocket League kickoff positions from GigaLearnCPP
        """
        Atomic.section("KICKOFF BEHAVIOR ANALYSIS", 7)


        # Generate kickoff states (base 109 dimensions)
        kickoff_states = generate_kickoff_states(num_kickoffs)

        with torch.no_grad():
            # Prepare inputs for each model
            base_obs_size = 109
            kickoff_tensor1 = torch.from_numpy(kickoff_states).to(self.device)
            kickoff_tensor2 = torch.from_numpy(kickoff_states).to(self.device)

            # Pad/crop to match each model's expected input
            if self.original_cfg.obs_size > base_obs_size:
                kickoff_tensor1 = F.pad(kickoff_tensor1, (0, self.original_cfg.obs_size - base_obs_size))
            elif self.original_cfg.obs_size < base_obs_size:
                kickoff_tensor1 = kickoff_tensor1[:, :self.original_cfg.obs_size]

            if self.suspicious_cfg.obs_size > base_obs_size:
                kickoff_tensor2 = F.pad(kickoff_tensor2, (0, self.suspicious_cfg.obs_size - base_obs_size))
            elif self.suspicious_cfg.obs_size < base_obs_size:
                kickoff_tensor2 = kickoff_tensor2[:, :self.suspicious_cfg.obs_size]

            # Get predictions
            def get_probs(model_dict, x):
                if 'shared_head' in model_dict:
                    x = model_dict['shared_head'](x)
                if 'policy' in model_dict:
                    x = model_dict['policy'](x)
                return F.softmax(x, dim=-1)

            probs1 = get_probs(self.original_model, kickoff_tensor1)
            probs2 = get_probs(self.suspicious_model, kickoff_tensor2)
            
            # Compare action distributions on kickoffs
            actions1 = torch.argmax(probs1, dim=-1).cpu().numpy()
            actions2 = torch.argmax(probs2, dim=-1).cpu().numpy()
            
            # Action agreement
            agreement = np.mean(actions1 == actions2)
            
            # KL divergence on probability distributions
            kl_divs = []
            for p1, p2 in zip(probs1, probs2):
                min_actions = min(len(p1), len(p2))
                p1_trim = p1[:min_actions]
                p2_trim = p2[:min_actions]
                kl = F.kl_div(p2_trim.log(), p1_trim, reduction='sum').item()
                kl_divs.append(kl)
            
            avg_kl = np.mean(kl_divs)
            
            # Analyze action diversity
            unique_actions1 = len(np.unique(actions1))
            unique_actions2 = len(np.unique(actions2))
        
        self.results['kickoff_analysis'] = {
            'action_agreement': float(agreement),
            'avg_kl_divergence': float(avg_kl),
            'action_diversity_model1': int(unique_actions1),
            'action_diversity_model2': int(unique_actions2),
            'num_kickoffs_tested': num_kickoffs
        }
        
        if agreement > 0.8:
            verdict = "[CRIT] IDENTICAL KICKOFF BEHAVIOR"
            c = Atomic.RED
        elif agreement > 0.6:
            verdict = "[WARN] SIMILAR KICKOFF DECISIONS"
            c = Atomic.YELLOW
        else:
            verdict = "[SAFE] DIFFERENT KICKOFF STRATEGIES"
            c = Atomic.GREEN
        
        Atomic.kv("Action Agreement", f"{agreement:.4f}", c)
        Atomic.kv("KL Divergence", f"{avg_kl:.4f}", Atomic.WHITE)
        Atomic.kv("Verdict", verdict, c)
        
        self.results['kickoff_analysis']['verdict'] = verdict
        return self.results['kickoff_analysis']
    
    def analyze_eigenvalue_spectrum(self) -> Dict:
        """
        [NEW METHOD 2/8] Compare eigenvalue spectra using Entropy and Effective Rank.
        Old method (Correlation) was flawed because all nets look the same.
        This method checks 'Information Density' and 'Complexity' instead.
        """
        Atomic.section("EIGENVALUE SPECTRUM (ENTROPY)", 8)
        
        from scipy.linalg import svd
        
        def get_weight_matrices(model_dict):
            matrices = []
            for key in ['shared_head', 'policy']:
                if key in model_dict:
                    for name, param in model_dict[key].named_parameters():
                        # Only grab 2D weights (Linear layers), skip biases/norms
                        if 'weight' in name and param.ndim == 2:
                            matrices.append(param.detach().cpu().numpy())
            return matrices
        
        matrices1 = get_weight_matrices(self.original_model)
        matrices2 = get_weight_matrices(self.suspicious_model)
        
        # If architecture is totally different, we can't compare layer-by-layer
        if len(matrices1) != len(matrices2):
            Atomic.log("Architecture mismatch - skipping spectral comparison", "WARN")
            self.results['eigenvalue_spectrum'] = {'status': 'skipped_arch_mismatch'}
            return self.results['eigenvalue_spectrum']
            
        entropy_diffs = []
        rank_diffs = []
        
        for i, (m1, m2) in enumerate(zip(matrices1, matrices2)):
            if m1.shape != m2.shape: continue
            
            try:
                # 1. Compute Singular Values (The 'Energy' of the layer)
                # svd returns sorted singular values
                _, s1, _ = svd(m1, full_matrices=False)
                _, s2, _ = svd(m2, full_matrices=False)
                
                # 2. Compute Spectral Entropy (The 'IQ' of the layer)
                # Normalize sum to 1 to treat as probabilities
                p1 = s1 / np.sum(s1)
                p2 = s2 / np.sum(s2)
                
                # Entropy = -sum(p * log(p))
                h1 = -np.sum(p1 * np.log(p1 + 1e-12))
                h2 = -np.sum(p2 * np.log(p2 + 1e-12))
                
                # Check absolute difference in entropy
                entropy_diffs.append(abs(h1 - h2))
                
                # 3. Compute Effective Rank (How many dims strictly needed?)
                # Count how many singular values are needed to explain 99% of variance
                def effective_rank(s):
                    cum_energy = np.cumsum(s) / np.sum(s)
                    return np.searchsorted(cum_energy, 0.99) + 1
                
                r1 = effective_rank(s1)
                r2 = effective_rank(s2)
                
                # Relative rank difference
                rank_diff = abs(r1 - r2) / max(r1, r2)
                rank_diffs.append(rank_diff)
                
            except Exception as e:
                pass # Skip broken layers
        
        # Average differences (Lower is more suspicious)
        avg_entropy_diff = np.mean(entropy_diffs) if entropy_diffs else 1.0
        avg_rank_diff = np.mean(rank_diffs) if rank_diffs else 1.0
        
        self.results['eigenvalue_spectrum'] = {
            'avg_entropy_diff': float(avg_entropy_diff),
            'avg_rank_diff': float(avg_rank_diff)
        }
        
        # VERDICT LOGIC
        # If entropy diff is tiny (< 0.05), it means the layers hold the exact same amount of information.
        # This is very hard to achieve by accident.
        
        if avg_entropy_diff < 0.02 and avg_rank_diff < 0.02:
            verdict = "[CRIT] IDENTICAL COMPLEXITY (TL)"
            c = Atomic.RED
        elif avg_entropy_diff < 0.1:
            verdict = "[WARN] SIMILAR COMPLEXITY"
            c = Atomic.YELLOW
        else:
            verdict = "[SAFE] DISTINCT INFORMATION"
            c = Atomic.GREEN
        
        # Invert the metric for display so "Higher" = "More Similar" (easier for users to read)
        display_score = max(0, 1.0 - avg_entropy_diff)
        
        Atomic.kv("Entropy Similarity", f"{display_score:.4f}", c)
        Atomic.kv("Rank Diff", f"{avg_rank_diff:.4f}", Atomic.WHITE)
        Atomic.kv("Verdict", verdict, c)
        
        self.results['eigenvalue_spectrum']['verdict'] = verdict
        return self.results['eigenvalue_spectrum']
    
    def analyze_transfer_learning_signature(self) -> Dict:
        """
        [NEW METHOD 3/8] Detect specific transfer learning patterns
        Early layers similar + late layers different = classic TL signature
        """
        Atomic.section("TRANSFER LEARNING SIGNATURE", 9)
        
        def get_layer_weights(model_dict):
            weights = []
            for key in ['shared_head', 'policy']:
                if key in model_dict:
                    for name, param in model_dict[key].named_parameters():
                        if 'weight' in name:
                            weights.append(param.detach().cpu().numpy())
            return weights
        
        weights1 = get_layer_weights(self.original_model)
        weights2 = get_layer_weights(self.suspicious_model)
        
        if len(weights1) != len(weights2):
            self.results['transfer_signature'] = {
                'signature_detected': False,
                'reason': 'different_architectures'
            }
            Atomic.log("Cannot detect signature: different architectures", "WARN")
            return self.results['transfer_signature']
        
        # Compute similarity for each layer
        layer_similarities = []
        for w1, w2 in zip(weights1, weights2):
            if w1.shape == w2.shape:
                w1_flat = w1.flatten()
                w2_flat = w2.flatten()
                sim = 1 - cosine(w1_flat, w2_flat)
                layer_similarities.append(sim)
            else:
                layer_similarities.append(0)
        
        self.results['transfer_signature'] = {
            'layer_similarities': [float(s) for s in layer_similarities]
        }
        
        # Check for transfer learning pattern
        if len(layer_similarities) >= 3:
            early_avg = np.mean(layer_similarities[:len(layer_similarities)//2])
            late_avg = np.mean(layer_similarities[len(layer_similarities)//2:])
            
            # Transfer learning signature: early layers similar, later different
            if early_avg > 0.8 and late_avg < 0.6:
                self.results['transfer_signature']['signature_detected'] = True
                self.results['transfer_signature']['pattern'] = 'CLASSIC_TL: early_frozen_late_trained'
                verdict = "[CRIT] CLASSIC TRANSFER LEARNING PATTERN"
                c = Atomic.RED
            elif early_avg > 0.9 and late_avg > 0.9:
                self.results['transfer_signature']['signature_detected'] = True
                self.results['transfer_signature']['pattern'] = 'FULL_COPY: entire_model_stolen'
                verdict = "[CRIT] FULL MODEL COPY"
                c = Atomic.RED
            else:
                self.results['transfer_signature']['signature_detected'] = False
                verdict = "[SAFE] NO TL SIGNATURE"
                c = Atomic.GREEN
            
            self.results['transfer_signature']['early_similarity'] = float(early_avg)
            self.results['transfer_signature']['late_similarity'] = float(late_avg)
            
            Atomic.kv("Early Layers", f"{early_avg:.4f}", Atomic.WHITE)
            Atomic.kv("Late Layers", f"{late_avg:.4f}", Atomic.WHITE)
            Atomic.kv("Verdict", verdict, c)
        else:
            self.results['transfer_signature']['signature_detected'] = False
            self.results['transfer_signature']['reason'] = 'insufficient_layers'
            Atomic.log("Not enough layers for signature detection", "WARN")
        
        return self.results['transfer_signature']
    
    def analyze_decision_boundaries(self, num_samples: int = 200) -> Dict:
        """
        [NEW METHOD 4/8] Compare decision boundaries using perturbations
        """
        Atomic.section("DECISION BOUNDARY ANALYSIS", 10)
        
        obs_size = min(self.original_cfg.obs_size, self.suspicious_cfg.obs_size)
        base_states = torch.randn(num_samples, obs_size).to(self.device)
        
        boundary_agreements = []
        
        with torch.no_grad():
            def get_pred(model_dict, x):
                if 'shared_head' in model_dict:
                    x = model_dict['shared_head'](x)
                if 'policy' in model_dict:
                    x = model_dict['policy'](x)
                return torch.argmax(x, dim=-1)
            
            # Test perturbations around each state
            for i in range(min(50, num_samples)):
                base = base_states[i:i+1]
                
                # Generate perturbations
                perturbations = []
                for _ in range(10):
                    noise = torch.randn_like(base) * 0.1
                    perturbations.append(base + noise)
                
                perturbations = torch.cat(perturbations, dim=0)
                
                # Pad/crop for each model
                if self.original_cfg.obs_size > obs_size:
                    p1 = F.pad(perturbations, (0, self.original_cfg.obs_size - obs_size))
                else:
                    p1 = perturbations[:, :self.original_cfg.obs_size]
                
                if self.suspicious_cfg.obs_size > obs_size:
                    p2 = F.pad(perturbations, (0, self.suspicious_cfg.obs_size - obs_size))
                else:
                    p2 = perturbations[:, :self.suspicious_cfg.obs_size]
                
                perturbed_pred1 = get_pred(self.original_model, p1)
                perturbed_pred2 = get_pred(self.suspicious_model, p2)
                
                agreement = (perturbed_pred1 == perturbed_pred2).float().mean().item()
                boundary_agreements.append(agreement)
        
        avg_agreement = np.mean(boundary_agreements) if boundary_agreements else 0
        std_agreement = np.std(boundary_agreements) if boundary_agreements else 0
        
        self.results['decision_boundaries'] = {
            'boundary_agreement': float(avg_agreement),
            'boundary_std': float(std_agreement)
        }
        
        if avg_agreement > 0.8:
            verdict = "[CRIT] IDENTICAL DECISION BOUNDARIES"
            c = Atomic.RED
        elif avg_agreement > 0.6:
            verdict = "[WARN] SIMILAR BOUNDARIES"
            c = Atomic.YELLOW
        else:
            verdict = "[SAFE] DIFFERENT BOUNDARIES"
            c = Atomic.GREEN
        
        Atomic.kv("Boundary Agreement", f"{avg_agreement:.4f}", c)
        Atomic.kv("Std Dev", f"{std_agreement:.4f}", Atomic.WHITE)
        Atomic.kv("Verdict", verdict, c)
        
        self.results['decision_boundaries']['verdict'] = verdict
        return self.results['decision_boundaries']
    
    def analyze_temporal_consistency(self, num_sequences: int = 20) -> Dict:
        """
        [NEW METHOD 5/8] Analyze how decisions evolve over game sequences
        """
        Atomic.section("TEMPORAL CONSISTENCY", 11)
        
        sequence_length = 10
        sequence_agreements = []
        
        with torch.no_grad():
            for _ in range(num_sequences):
                # Start with kickoff
                kickoff = generate_kickoff_states(1)[0]
                
                # Evolve state
                sequence = [kickoff]
                for _ in range(sequence_length - 1):
                    next_state = sequence[-1].copy()
                    # Update ball position
                    next_state[0:3] += next_state[3:6] * 0.1
                    # Add noise
                    next_state += np.random.randn(109) * 0.05
                    next_state = np.clip(next_state, -2, 2)
                    sequence.append(next_state)
                
                sequence_tensor = torch.from_numpy(np.array(sequence)).float().to(self.device)
                
                # Pad/crop for each model
                if self.original_cfg.obs_size > 109:
                    seq1 = F.pad(sequence_tensor, (0, self.original_cfg.obs_size - 109))
                else:
                    seq1 = sequence_tensor[:, :self.original_cfg.obs_size]
                
                if self.suspicious_cfg.obs_size > 109:
                    seq2 = F.pad(sequence_tensor, (0, self.suspicious_cfg.obs_size - 109))
                else:
                    seq2 = sequence_tensor[:, :self.suspicious_cfg.obs_size]
                
                # Get action sequences
                def get_action_sequence(model_dict, states):
                    actions = []
                    for state in states:
                        x = state.unsqueeze(0)
                        if 'shared_head' in model_dict:
                            x = model_dict['shared_head'](x)
                        if 'policy' in model_dict:
                            x = model_dict['policy'](x)
                        action = torch.argmax(x, dim=-1).item()
                        actions.append(action)
                    return actions
                
                actions1 = get_action_sequence(self.original_model, seq1)
                actions2 = get_action_sequence(self.suspicious_model, seq2)
                
                agreement = sum(a1 == a2 for a1, a2 in zip(actions1, actions2)) / sequence_length
                sequence_agreements.append(agreement)
        
        avg_agreement = np.mean(sequence_agreements) if sequence_agreements else 0
        std_agreement = np.std(sequence_agreements) if sequence_agreements else 0
        
        self.results['temporal_consistency'] = {
            'temporal_agreement': float(avg_agreement),
            'temporal_std': float(std_agreement)
        }
        
        if avg_agreement > 0.75:
            verdict = "[CRIT] IDENTICAL TEMPORAL BEHAVIOR"
            c = Atomic.RED
        elif avg_agreement > 0.6:
            verdict = "[WARN] SIMILAR TEMPORAL PATTERNS"
            c = Atomic.YELLOW
        else:
            verdict = "[SAFE] DIFFERENT TEMPORAL BEHAVIOR"
            c = Atomic.GREEN
        
        Atomic.kv("Temporal Agreement", f"{avg_agreement:.4f}", c)
        Atomic.kv("Std Dev", f"{std_agreement:.4f}", Atomic.WHITE)
        Atomic.kv("Verdict", verdict, c)
        
        self.results['temporal_consistency']['verdict'] = verdict
        return self.results['temporal_consistency']
    
    # ==================== END ADVANCED METHODS ====================
    
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
        
        # ==================== NEW ADVANCED METHODS ====================
        try:
            self.analyze_kickoff_behavior()
        except Exception as e:
            Atomic.log(f"Kickoff analysis failed: {e}", "CRIT")
        
        try:
            self.analyze_eigenvalue_spectrum()
        except Exception as e:
            Atomic.log(f"Eigenvalue analysis failed: {e}", "CRIT")
        
        try:
            self.analyze_transfer_learning_signature()
        except Exception as e:
            Atomic.log(f"TL signature analysis failed: {e}", "CRIT")
        
        try:
            self.analyze_decision_boundaries()
        except Exception as e:
            Atomic.log(f"Decision boundary analysis failed: {e}", "CRIT")
        
        try:
            self.analyze_temporal_consistency()
        except Exception as e:
            Atomic.log(f"Temporal consistency analysis failed: {e}", "CRIT")
        # ==================== END NEW METHODS ====================
        
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
