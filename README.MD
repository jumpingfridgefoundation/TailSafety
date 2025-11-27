# Tail Safety Engine, alpha Experimental

**Warning: This is not a production-ready TTS engine. This is what happens when we get bored and decide to synthesize human speech using raw math in 48 hours.**

##  About
The Tail Safety Engine is a **Formant Synthesizer**. Unlike modern AI that uses deep learning or concatenative synthesis (gluing recorded samples together), this engine uses `numpy` and `scipy` to physically simulate the resonances (formants) of the human vocal tract.

There are **zero** audio recordings in this repo. Every sound you hear is a generated sine wave filtered through a digital approximation of a throat and mouth.

It is lightweight, 100% open-source, and arguably distinct.

##  Features
*   **Dynamic Prosody:** The engine simulates breathing, pitch drift, and tempo variance so it doesn't sound completely dead inside.
*   **Multilingual Support (Technically):**
    *   **English:** Functional.
    *   **Russian:** Experimental.
    *   **Arabic:** *Highly* experimental (it tries its best).

##  Known Limitations (Read This!)
1.  **NO NUMBERS:** The text parser is very simple. If you type `1995`, it will crash or ignore it. Type `nineteen ninety five` instead.
2.  **Arabic is a Mess:** The diacritic handling and phoneme mapping for Arabic are barely holding together. It sounds like a robot trying to speak Arabic while underwater.
3.  **Stability:** This was built in two days. It handles errors gracefully-ish, but don't use this for mission-critical accessibility tools.

##  Installation

You need Python installed. Then, grab the dependencies:

```bash
pip install numpy scipy sounddevice nltk
```

*Optional:* If you want slightly better Arabic processing, install Mishkal:
```bash
pip install mishkal
```

##  Usage

1.  Clone the repo.
2.  Run the main file:
    ```bash
    python main.py
    ```
3.  Type your text and hit enter.

**Example Input:**
> "Hello world. I am a formant synthesizer."

**To Quit:**
Type `exit` and hit enter.

##  How It Works
*   **`src/g2p.py`**: Converts text into phonemes (using CMU Dict for English, and custom rule-based maps for RU/AR).
*   **`src/engine.py`**: The core logic. It generates tracks for Formant Frequencies (F1-F4), Pitch, and Amplitude.
*   **`src/config.py`**: Contains the "DNA" of the voice—the exact frequency values for every vowel and consonant.

##  License & Terms
**The Jumping Fridge Foundation** grants you the liberty to do whatever you want with this code: modify it, break it, use it for screen readers, or turn it into a beatboxing bot.

**You cannot sell this.** This is free software meant for educational and experimental purposes.

