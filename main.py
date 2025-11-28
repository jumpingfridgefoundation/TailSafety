# Import the engine from the source module
from src.engine import TailSafetyEngine
from src.voice_loader import load_voices_from_directory, get_voice_list, print_voices, get_voice_by_name
from src import config
import os

if __name__ == "__main__":
    # Initialize the TTS engine
    print("TailSafety")
    print("use /voices To see a list of available voices")
    print("use The word voice followed by a space and then the voice number To switch to it")
    
    # Find voices directory (check in current directory and parent)
    voices_dir = None
    for path in ['voices', '../voices', './voices']:
        if os.path.isdir(path):
            voices_dir = path
            break
    
    # Load voices from directory
    if voices_dir:
        print(f"\nLoading voices from: {os.path.abspath(voices_dir)}/")
        loaded_voices = load_voices_from_directory(voices_dir)
    else:
        print("\nWarning: voices/ directory not found. Using fallback voices from config.")
        loaded_voices = {}
    
    # Use loaded voices if available, otherwise fallback to config profiles
    voice_profiles = loaded_voices if loaded_voices else config.VOICE_PROFILES.copy()
    
    # Display available voices
    print_voices(voice_profiles)
    
    # Default voice - use first available or raise error if none are available
    voice_list = get_voice_list(voice_profiles)
    if voice_list:
        current_voice_key = list(voice_profiles.keys())[0]
        current_voice_name = voice_profiles[current_voice_key]['name']
    else:
        raise RuntimeError("No voice profiles available. Check your 'voices' directory and config.py.")
    
    tts = TailSafetyEngine(voice_profile=voice_profiles[current_voice_key])
    print(f"\n✓ Ready. Current voice: {current_voice_name}")
    print("Version: 46")
    
    # Main interaction loop for text-to-speech processing
    while True:
        try:
            user_input = input("\n📝 Text (or /voices, voice <num/name>, /log, exit): ").strip()
            
            if user_input.lower() == 'exit':
                print("Goodbye!")
                break
            
            if user_input.lower() == '/voices':
                print_voices(voice_profiles, current_voice_key)
                continue
            
            
            
            if user_input.lower().startswith('voice '):
                try:
                    parts = user_input.split(maxsplit=1)
                    if len(parts) > 1:
                        voice_input = parts[1].strip()
                        
                        # Try to parse as number first
                        try:
                            voice_num = int(voice_input) - 1
                            voice_list = get_voice_list(voice_profiles)
                            if 0 <= voice_num < len(voice_list):
                                current_voice_key = voice_list[voice_num]['key']
                                current_voice_name = voice_profiles[current_voice_key]['name']
                                tts = TailSafetyEngine(voice_profile=voice_profiles[current_voice_key])
                                print(f"✓ Switched to: {current_voice_name}")
                            else:
                                print(f"Please enter a number between 1 and {len(voice_list)}")
                        except ValueError:
                            # Try to match by name
                            matched_key, matched_data = get_voice_by_name(voice_profiles, voice_input)
                            if matched_key:
                                current_voice_key = matched_key
                                current_voice_name = matched_data['name']
                                tts = TailSafetyEngine(voice_profile=matched_data)
                                print(f"✓ Switched to: {current_voice_name}")
                            else:
                                print(f"Voice '{voice_input}' not found. Use 'list' to see available voices.")
                except Exception as e:
                    print(f"Usage: voice <number or name>")
                continue
            
            if user_input:
                print(f"🔊 Speaking as {voice_profiles[current_voice_key]['name']}...")
                tts.speak(user_input)
                print("✓ Done")
        except Exception as e:
            # Handle any runtime errors gracefully
            print(f"Error: {e}")