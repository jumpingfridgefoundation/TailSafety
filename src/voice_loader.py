"""
Voice Loader - Dynamically loads voice profiles from the voices/ folder structure
"""

import os
import json
from pathlib import Path


def load_voices_from_directory(voices_dir="voices"):
    """
    Loads all voices from Python modules in the voices/ directory.
    Each voice folder should contain a .py file with VOICE_PROFILE dictionary.
    """
    import importlib.util
    import sys
    voices = {}
    voices_path = Path(voices_dir)
    
    if not voices_path.exists():
        print(f"Warning: {voices_dir} directory not found. Using fallback voices.")
        return {}
    
    for voice_folder in voices_path.iterdir():
        if not voice_folder.is_dir():
            continue
        
        # Find a .py file in the voice folder
        py_files = list(voice_folder.glob("*.py"))
        if not py_files:
            print(f"Warning: No Python module found in {voice_folder.name}")
            continue
        
        py_file = py_files[0]
        module_name = f"voices.{voice_folder.name}.{py_file.stem}"
        
        try:
            spec = importlib.util.spec_from_file_location(module_name, str(py_file))
            mod = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = mod
            spec.loader.exec_module(mod)
            
            profile = getattr(mod, "VOICE_PROFILE", None)
            if profile:
                voice_key = profile.get('name', voice_folder.name)
                voices[voice_key] = profile
                print(f"[OK] Loaded voice: {voice_key} from {voice_folder.name}/")
            else:
                print(f"Warning: No VOICE_PROFILE in {py_file}")
        except Exception as e:
            print(f"Error loading {py_file}: {e}")
    
    return voices


def get_voice_list(voices_dict):
    """
    Get a formatted list of available voices.
    
    Args:
        voices_dict: Dictionary of voice profiles
        
    Returns:
        list: List of voices sorted by name
    """
    voice_list = []
    for voice_key, voice_data in voices_dict.items():
        voice_list.append({
            'key': voice_key,
            'name': voice_data['name'],
            'gender': voice_data['gender'],
            'accent': voice_data['accent'],
            'description': voice_data.get('description', ''),
        })
    
    # Sort by real name
    voice_list.sort(key=lambda x: x['name'])
    return voice_list


def get_voice_by_name(voices_dict, name):
    """
    Get a voice by its real name.
    
    Args:
        voices_dict: Dictionary of voice profiles
        name: Real name of the voice (e.g., "Sarah", "James")
        
    Returns:
        tuple: (voice_key, voice_data) or (None, None) if not found
    """
    for voice_key, voice_data in voices_dict.items():
        if voice_data['name'].lower() == name.lower():
            return voice_key, voice_data
    
    return None, None


def print_voices(voices_dict, current_voice_key=None):
    """
    Print formatted voice list to console.
    
    Args:
        voices_dict: Dictionary of voice profiles
        current_voice_key: Key of currently selected voice (optional)
    """
    if not voices_dict:
        print("No voices available.")
        return
    
    print("\n--- Available Voices ---")
    voice_list = get_voice_list(voices_dict)
    
    for i, voice in enumerate(voice_list, 1):
        current_marker = " [CURRENT]" if voice['key'] == current_voice_key else ""
        print(f"{i}. {voice['name']} ({voice['gender']}, {voice['accent']}){current_marker}")
        if voice['description']:
            print(f"   {voice['description']}")
