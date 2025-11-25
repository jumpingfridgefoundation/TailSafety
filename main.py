# Import the TTS safety engine from the source module
from src.engine import TailSafetyEngine

if __name__ == "__main__":
    # Initialize the TTS engine with safety features
    print("\n SAFETY ENGINE V40")
    print("Initializing...")
    tts = TailSafetyEngine()
    print("Ready.")
    print("Features: English/Russian/Arabic, Dynamic Prosody, Tempo Drift.")
    
    # Main interaction loop for text-to-speech processing
    while True:
        try:
            t = input("\nText: ").strip()
            if t.lower() == 'exit': break
            if t: tts.speak(t)
        except Exception as e:
            # Handle any runtime errors gracefully
            print(f"Error: {e}")