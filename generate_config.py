"""
Auto-Configuration Generator for Bot Analysis

This script automatically generates JSON configuration files from existing bot directories.
It detects layer sizes, obs sizes, action sizes, etc. by analyzing the bot's Python files.

Usage:
    python generate_config.py /path/to/bot/directory output.json
"""

import os
import sys
import re
import json
import argparse
from pathlib import Path
import ast


class BotConfigGenerator:
    """Automatically generates bot configuration from bot directory"""
    
    def __init__(self, bot_dir: str):
        self.bot_dir = Path(bot_dir)
        if not self.bot_dir.exists():
            raise FileNotFoundError(f"Bot directory not found: {bot_dir}")
        
        self.config = {
            "bot_name": self.bot_dir.name,
            "model_path": "",
            "architecture": {
                "shared_head_layers": [],
                "policy_layers": [],
                "activation": "relu",
                "layer_norm": True
            },
            "observation": {
                "obs_size": 0,
                "obs_builder_path": "",
                "obs_builder_class": "AdvancedObs"
            },
            "action_parser": {
                "action_size": 0,
                "action_parser_path": "",
                "action_parser_class": "LookupAction",
                "action_bins": [[-1, 0, 1], [-1, 0, 1], [-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]
            },
            "discrete_policy": {
                "discrete_policy_path": "",
                "discrete_policy_class": "DiscreteFF"
            },
            "additional_paths": [],
            "metadata": {
                "tournament": "",
                "author": "",
                "training_steps": 0,
                "notes": "Auto-generated configuration"
            }
        }
    
    def find_file(self, pattern: str) -> Path:
        """Find file matching pattern in bot directory"""
        for file in self.bot_dir.rglob(pattern):
            return file
        return None
    
    def extract_variable_value(self, file_path: Path, var_name: str):
        """Extract variable value from Python file"""
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            
            # Try to find variable assignment
            pattern = rf'{var_name}\s*=\s*(.+)'
            match = re.search(pattern, content)
            
            if match:
                value_str = match.group(1).strip()
                # Try to evaluate as Python literal
                try:
                    return ast.literal_eval(value_str)
                except:
                    return value_str
        except Exception as e:
            print(f"  ⚠️  Error reading {file_path}: {e}")
        
        return None
    
    def analyze_agent_py(self):
        """Analyze agent.py to extract architecture info"""
        agent_file = self.find_file("agent.py")
        if not agent_file:
            print("  ⚠️  agent.py not found")
            return
        
        print(f"  ✓ Found agent.py: {agent_file}")
        
        with open(agent_file, 'r') as f:
            content = f.read()
        
        # Extract OBS_SIZE
        obs_size = self.extract_variable_value(agent_file, "OBS_SIZE")
        if obs_size:
            self.config['observation']['obs_size'] = obs_size
            print(f"    - OBS_SIZE: {obs_size}")
        
        # Extract SHARED_LAYER_SIZES
        shared_layers = self.extract_variable_value(agent_file, "SHARED_LAYER_SIZES")
        if shared_layers:
            self.config['architecture']['shared_head_layers'] = shared_layers
            print(f"    - SHARED_LAYER_SIZES: {shared_layers}")
        
        # Extract POLICY_LAYER_SIZES
        policy_layers = self.extract_variable_value(agent_file, "POLICY_LAYER_SIZES")
        if policy_layers:
            self.config['architecture']['policy_layers'] = policy_layers
            print(f"    - POLICY_LAYER_SIZES: {policy_layers}")
    
    def analyze_obs_py(self):
        """Analyze obs.py to extract observation builder info"""
        obs_file = self.find_file("obs.py")
        if not obs_file:
            print("  ⚠️  obs.py not found")
            return
        
        print(f"  ✓ Found obs.py: {obs_file}")
        self.config['observation']['obs_builder_path'] = str(obs_file)
        
        with open(obs_file, 'r') as f:
            content = f.read()
        
        # Find observation builder class name
        class_match = re.search(r'class\s+(\w+).*:', content)
        if class_match:
            class_name = class_match.group(1)
            self.config['observation']['obs_builder_class'] = class_name
            print(f"    - Obs Builder Class: {class_name}")
    
    def analyze_action_parser(self):
        """
        Analyze action parser file.
        
        updates: now uses black magic (dynamic imports) to try and actually run the code
        because you people keep writing procedural action generators that regex can't read.
        """
        # Try common names
        for name in ["your_act.py", "action_parser.py", "actions.py", "discrete_act.py"]:
            action_file = self.find_file(name)
            if action_file:
                break
        
        if not action_file:
            print("  ⚠️  Action parser file not found. good luck.")
            return
        
        print(f"  ✓ Found action parser: {action_file}")
        self.config['action_parser']['action_parser_path'] = str(action_file)
        
        with open(action_file, 'r') as f:
            content = f.read()
        
        # Find action parser class name so we know what to instantiate later
        class_name = "LookupAction" # default fallback
        class_match = re.search(r'class\s+(\w+).*:', content)
        if class_match:
            class_name = class_match.group(1)
            self.config['action_parser']['action_parser_class'] = class_name
            print(f"    - Action Parser Class: {class_name}")

        # --- THE "I GIVE UP" SECTION ---
        # attempt to dynamically import the file and run it. 
        # yes, this is security nightmare. no, i don't care anymore.
        try:
            import importlib.util
            
            # tell python to pretend this random file is a module
            spec = importlib.util.spec_from_file_location("dynamic_action_parser", action_file)
            module = importlib.util.module_from_spec(spec)
            
            # hack to make relative imports inside the bot work (maybe)
            sys.modules["dynamic_action_parser"] = module 
            
            # execute the module. if this crashes because you don't have numpy installed, 
            # that is between you and god.
            spec.loader.exec_module(module)
            
            if hasattr(module, class_name):
                cls = getattr(module, class_name)
                
                # attempt to instantiate. pray it takes no arguments in __init__
                # if your init requires arguments, i am so sorry but you are on your own.
                print(f"    ... attempting to instantiate {class_name} to check size ...")
                instance = cls()
                
                size = 0
                # check for standard methods/attributes
                if hasattr(instance, 'get_action_space_size'):
                    size = instance.get_action_space_size()
                elif hasattr(instance, '_lookup_table'):
                    # support for that specific weird C++ port logic you pasted
                    size = len(instance._lookup_table)
                elif hasattr(instance, 'actions'):
                    size = len(instance.actions)
                
                if size > 0:
                    self.config['action_parser']['action_size'] = int(size)
                    print(f"    ✓ calculated action size via execution: {size}")
                    return # we are done here, thank the lord
                    
        except ImportError as e:
            print(f"    ⚠️  Failed to import action parser (missing libs?): {e}")
        except Exception as e:
            print(f"    ⚠️  Dynamic analysis failed (it was a long shot anyway): {e}")

        # --- FALLBACK TO REGEX GUESSING ---
        # if we reach here, the dynamic import failed. back to guessing.
        
        # Try to count actions in lookup table via bins definition
        if 'make_lookup_table' in content or '_lookup_table' in content:
            # Try to estimate action size from bins
            bins_match = re.search(r'bins\s*=\s*\[([^\]]+)\]', content)
            if bins_match:
                try:
                    # Rough estimate based on standard LookupAction
                    self.config['action_parser']['action_size'] = 90
                    print(f"    - Estimated action size: 90 (standard LookupAction)")
                except:
                    pass
            else:
                 print("    ⚠️  Could not determine action size. Please fill 'action_size' manually in json.")
    
    def analyze_discrete_policy(self):
        """Analyze discrete policy file"""
        policy_file = self.find_file("discrete_policy.py")
        if not policy_file:
            print("  ⚠️  discrete_policy.py not found")
            return
        
        print(f"  ✓ Found discrete_policy.py: {policy_file}")
        self.config['discrete_policy']['discrete_policy_path'] = str(policy_file)
        
        with open(policy_file, 'r') as f:
            content = f.read()
        
        # Find policy class
        class_match = re.search(r'class\s+(\w+).*nn\.Module.*:', content)
        if class_match:
            class_name = class_match.group(1)
            self.config['discrete_policy']['discrete_policy_class'] = class_name
            print(f"    - Discrete Policy Class: {class_name}")
    
    def find_model_files(self):
        """Find model files (SHARED_HEAD.lt, POLICY.lt)"""
        # Look for .lt or .LT files
        for ext in [".lt", ".LT"]:
            policy_file = self.find_file(f"POLICY{ext}")
            if policy_file:
                # Use parent directory as model path
                self.config['model_path'] = str(policy_file.parent)
                print(f"  ✓ Found model files in: {policy_file.parent}")
                return
        
        print("  ⚠️  Model files not found")
    
    def add_additional_paths(self):
        """Add bot directory and util subdirectory to paths"""
        self.config['additional_paths'].append(str(self.bot_dir))
        
        util_dir = self.bot_dir / "util"
        if util_dir.exists():
            self.config['additional_paths'].append(str(util_dir))
    
    def generate(self) -> dict:
        """Generate complete configuration"""
        print(f"\nAnalyzing bot directory: {self.bot_dir}")
        print("="*60)
        
        self.analyze_agent_py()
        self.analyze_obs_py()
        self.analyze_action_parser()
        self.analyze_discrete_policy()
        self.find_model_files()
        self.add_additional_paths()
        
        print("="*60)
        print("Configuration generated!")
        
        return self.config
    
    def save(self, output_path: str):
        """Save configuration to JSON file"""
        with open(output_path, 'w') as f:
            json.dump(self.config, f, indent=2)
        print(f"\n✅ Configuration saved to: {output_path}")
    
    def print_summary(self):
        """Print configuration summary"""
        print("\n" + "="*60)
        print("CONFIGURATION SUMMARY")
        print("="*60)
        print(f"Bot Name: {self.config['bot_name']}")
        print(f"Model Path: {self.config['model_path']}")
        print(f"Obs Size: {self.config['observation']['obs_size']}")
        print(f"Action Size: {self.config['action_parser']['action_size']}")
        print(f"Shared Layers: {self.config['architecture']['shared_head_layers']}")
        print(f"Policy Layers: {self.config['architecture']['policy_layers']}")
        print("="*60)


def main():
    parser = argparse.ArgumentParser(
        description='Auto-generate JSON configuration from bot directory'
    )
    parser.add_argument('bot_dir', help='Path to bot directory')
    parser.add_argument('output', help='Output JSON file path')
    parser.add_argument('--bot-name', help='Override bot name')
    
    args = parser.parse_args()
    
    # Generate configuration
    generator = BotConfigGenerator(args.bot_dir)
    config = generator.generate()
    
    # Override bot name if provided
    if args.bot_name:
        config['bot_name'] = args.bot_name
    
    # Print summary
    generator.print_summary()
    
    # Save
    generator.save(args.output)
    
    print("\n📝 Next steps:")
    print(f"  1. Review and edit {args.output} if needed")
    print(f"  2. Run analysis: python tl_detector.py original.json sus.json")


if __name__ == "__main__":
    main()
