
import numpy as np

# --- CONFIGURATION ---
SAMPLE_RATE = 48000
BLOCK_MS = 2.0 
BLOCK_SAMPLES = int(SAMPLE_RATE * BLOCK_MS / 1000)
BIT_DEPTH = np.float64

# --- VOICE PROFILES ---
# DEPRECATED: Voice profiles are now loaded from Python modules in voices/ folder
# This dict is kept for reference only and is no longer used by the engine
VOICE_PROFILES = {
    'default_female': {
        'name': 'Default Female',
        'gender': 'female',
        'accent': 'neutral',
        'base_pitch': 130.0,
        'formant_scale': 1.0,
        'duration_scale': 1.0,
        'noise_level': 0.35,
        'brightness': 0.0,  # 0 = neutral, +1 = brighter, -1 = darker
    },
    'deep_male': {
        'name': 'Deep Male',
        'gender': 'male',
        'accent': 'neutral',
        'base_pitch': 85.0,  # Lower pitch
        'formant_scale': 1.25,  # Lower formants
        'duration_scale': 0.95,  # Slightly faster
        'noise_level': 0.32,
        'brightness': -0.3,  # Darker tone
    },
    'bright_female': {
        'name': 'Bright Female',
        'gender': 'female',
        'accent': 'neutral',
        'base_pitch': 145.0,  # Higher pitch
        'formant_scale': 0.95,  # Slightly higher formants
        'duration_scale': 1.05,  # Slightly slower
        'noise_level': 0.38,
        'brightness': 0.4,  # Brighter tone
    },
    'british_male': {
        'name': 'British Male',
        'gender': 'male',
        'accent': 'british',
        'base_pitch': 95.0,
        'formant_scale': 1.15,
        'duration_scale': 1.1,  # More deliberate
        'noise_level': 0.30,
        'brightness': -0.15,
    },
    'american_female': {
        'name': 'American Female',
        'gender': 'female',
        'accent': 'american',
        'base_pitch': 135.0,
        'formant_scale': 1.0,
        'duration_scale': 1.0,
        'noise_level': 0.36,
        'brightness': 0.1,
    },
    'scottish_male': {
        'name': 'Scottish Male',
        'gender': 'male',
        'accent': 'scottish',
        'base_pitch': 100.0,
        'formant_scale': 1.2,
        'duration_scale': 0.92,  # Faster
        'noise_level': 0.34,
        'brightness': -0.2,
    },
    'irish_female': {
        'name': 'Irish Female',
        'gender': 'female',
        'accent': 'irish',
        'base_pitch': 140.0,
        'formant_scale': 1.05,
        'duration_scale': 1.08,  # Slightly more lyrical
        'noise_level': 0.37,
        'brightness': 0.2,
    },
    'australian_male': {
        'name': 'Australian Male',
        'gender': 'male',
        'accent': 'australian',
        'base_pitch': 110.0,
        'formant_scale': 1.1,
        'duration_scale': 0.98,
        'noise_level': 0.35,
        'brightness': 0.1,
    },
    'indian_female': {
        'name': 'Indian Female',
        'gender': 'female',
        'accent': 'indian',
        'base_pitch': 142.0,
        'formant_scale': 1.0,
        'duration_scale': 1.15,  # Slower, more melodic
        'noise_level': 0.33,
        'brightness': 0.25,
    },
    'french_male': {
        'name': 'French Male',
        'gender': 'male',
        'accent': 'french',
        'base_pitch': 105.0,
        'formant_scale': 1.08,
        'duration_scale': 1.12,  # More deliberate
        'noise_level': 0.31,
        'brightness': -0.1,
    },
    'spanish_female': {
        'name': 'Spanish Female',
        'gender': 'female',
        'accent': 'spanish',
        'base_pitch': 138.0,
        'formant_scale': 1.02,
        'duration_scale': 1.0,
        'noise_level': 0.36,
        'brightness': 0.15,
    },
    'german_male': {
        'name': 'German Male',
        'gender': 'male',
        'accent': 'german',
        'base_pitch': 92.0,
        'formant_scale': 1.3,  # Heavier, deeper
        'duration_scale': 1.15,  # More measured
        'noise_level': 0.29,
        'brightness': -0.35,
    },
}

# --- PHONEME DATA ---
# Format: [Base_Dur, F1, F2, F3, F4, Gain_dB, Type]
# Type: 0=Vow, 1=Fric, 2=Stop, 3=Pause, 4=VoicedFric, 5=Glide, 6=Vowel-Like
PHONEMES = {
    # VOWELS (improved formants for natural sound)
    'IY': [85,  270, 2250, 2890, 3500, -1, 0],   # /iː/ - fleece
    'IH': [65,  390, 1950, 2650, 3400, 0, 0],    # /ɪ/ - kit
    'EH': [85,  520, 1750, 2450, 3350, 0, 0],    # /ɛ/ - dress
    'AE': [105, 720, 1680, 2350, 3350, 1, 0],    # /æ/ - trap
    'AA': [95,  730, 1090, 2330, 3400, 2, 0],    # /ɑː/ - palm
    'AO': [95,  610, 920,  2350, 3300, 1, 0],    # /ɔː/ - lot/thought
    'OW': [105, 460, 920,  2250, 3250, 1, 5],    # /oʊ/ - goat (glide)
    'UH': [75,  430, 1150, 2250, 3300, 0, 0],    # /ʊ/ - foot
    'UW': [85,  330, 890,  2150, 3250, -1, 0],   # /uː/ - goose
    'AH': [75,  640, 1240, 2450, 3350, -1, 0],   # /ʌ/ - strut
    'ER': [105, 490, 1350, 1550, 3250, -1, 0],   # /ɜː/ - nurse
    'AX': [55,  520, 1560, 2450, 3350, -3, 0],   # /ə/ - schwa (reduced)
    'EY': [115, 460, 1950, 2450, 3350, 0, 5],    # /eɪ/ - face (glide)
    'AY': [125, 650, 1950, 2550, 3400, 1, 5],    # /aɪ/ - price (glide)
    'AW': [125, 700, 1150, 2350, 3350, 1, 5],    # /aʊ/ - mouth (glide)
    'OY': [125, 600, 950,  2250, 3350, 0, 5],    # /ɔɪ/ - choice (glide)

    # FRICATIVES (more realistic energy and spectral shape)
    'S':  [115, 0, 0, 0, 0, -9, 1],     # /s/ - sibilant, reduced harshness
    'SH': [115, 0, 0, 0, 0, -11, 1],    # /ʃ/ - softer sibilant
    'Z':  [105, 360, 1750, 2850, 3650, -9, 4],   # /z/ - voiced sibilant
    'ZH': [105, 360, 1550, 2450, 3450, -11, 4],  # /ʒ/ - voiced palato-alveolar
    'F':  [95,  0, 0, 0, 0, -14, 1],    # /f/ - softer
    'V':  [85,  310, 1450, 2450, 3450, -11, 4],  # /v/ - voiced labiodental
    'TH': [95,  0, 0, 0, 0, -17, 1],    # /θ/ - theta, very soft
    'DH': [75,  320, 1550, 2550, 3450, -14, 4],  # /ð/ - voiced theta
    'HH': [75,  0, 0, 0, 0, -19, 1],    # /h/ - voiceless, minimal energy

    # NASALS & LIQUIDS (more resonance)
    'M':  [85, 290, 1050, 2250, 3550, -4, 0],    # /m/ - more body
    'N':  [85, 290, 1750, 2700, 3550, -4, 0],    # /n/ - more presence
    'NG': [95, 290, 1250, 2450, 3550, -5, 0],    # /ŋ/ - velar nasal
    'L':  [95, 420, 1150, 3050, 3700, -1, 0],    # /l/ - bright, clear
    'R':  [95, 370, 1380, 1600, 3400, -1, 0],    # /r/ - retroflex quality
    'W':  [95, 320, 650,  2250, 3300, 0, 5],     # /w/ - labial glide
    'Y':  [95, 320, 2250, 3150, 3750, 0, 5],     # /j/ - palatal glide

    # STOPS (kept as no formant since they're silence + burst)
    'K': [0, 0, 0, 0, 0, 0, 2], 'G': [0, 0, 0, 0, 0, 0, 2],
    'P': [0, 0, 0, 0, 0, 0, 2], 'B': [0, 0, 0, 0, 0, 0, 2],
    'T': [0, 0, 0, 0, 0, 0, 2], 'D': [0, 0, 0, 0, 0, 0, 2],
    'CH': [0, 0, 0, 0, 0, 0, 2], 'JH': [0, 0, 0, 0, 0, 0, 2],

    # SPECIALS (Arabic/Russian with better formants)
    'KH': [115, 0, 0, 0, 0, -11, 1],    # Arabic kh - softer
    'GH': [105, 420, 1280, 2480, 3450, -9, 4],   # Arabic gh - voiced
    'Q': [0, 0, 0, 0, 0, 0, 2],         # Arabic q - emphatic stop
    'RR': [75, 420, 1450, 2050, 3550, -1, 0],    # Russian r - trilled
    'AIN': [105, 820, 1380, 2580, 3550, -1, 4],  # Arabic ain - voiced pharyngeal
    'H_AR': [95, 0, 0, 0, 0, -13, 1],   # Arabic h - soft
    'S_AR': [115, 0, 0, 0, 0, -9, 1],   # Arabic s - pharyngeal
    'D_AR': [0, 0, 0, 0, 0, 0, 2],      # Arabic d - emphatic stop
    'T_AR': [0, 0, 0, 0, 0, 0, 2],      # Arabic t - emphatic stop
    'Z_AR': [85, 0, 0, 0, 0, -14, 4],   # Arabic z - pharyngeal voiced

    'PAUSE': [0, 0, 0, 0, 0, 0, 3],
    'BREATH': [600, 0, 0, 0, 0, 0, 3],
    'END_OF_STREAM': [3000, 0, 0, 0, 0, 0, 3]
}

DIPHTHONG_MAP = {
    'AY': ('AA', 'IY'), 'EY': ('EH', 'IY'), 'OY': ('AO', 'IY'),
    'AW': ('AA', 'UW'), 'OW': ('AO', 'UW')
}

PLOSIVE_DATA = {
    'G':  {'cl':50, 'burst':1500, 'vb':0.90, 'loc_f2':1200, 'loc_f3':2400, 'asp':None, 'b_db':-20},
    'K':  {'cl':60, 'burst':1800, 'vb':0.0, 'loc_f2':1200, 'loc_f3':2400, 'asp':'H', 'b_db':-10},
    'D':  {'cl':40, 'burst':3500, 'vb':0.90, 'loc_f2':1800, 'loc_f3':2800, 'asp':None, 'b_db':-18},
    'T':  {'cl':50, 'burst':3800, 'vb':0.0, 'loc_f2':1800, 'loc_f3':2800, 'asp':'S', 'b_db':-10},
    'B':  {'cl':45, 'burst':700,  'vb':0.90, 'loc_f2':800,  'loc_f3':2300, 'asp':None, 'b_db':-20},
    'P':  {'cl':55, 'burst':700,  'vb':0.0, 'loc_f2':800,  'loc_f3':2300, 'asp':'H', 'b_db':-12},
    'JH': {'cl':45, 'burst':3500, 'vb':0.90, 'loc_f2':1800, 'loc_f3':2600, 'asp':'ZH', 'b_db':-15},
    'CH': {'cl':55, 'burst':4000, 'vb':0.0, 'loc_f2':1800, 'loc_f3':2600, 'asp':'SH_HARD', 'b_db':-12},
    'Q':    {'cl':70, 'burst':1000, 'vb':0.0, 'loc_f2':900,  'loc_f3':2400, 'asp':'H', 'b_db':-10},
    'D_AR': {'cl':55, 'burst':3000, 'vb':0.90, 'loc_f2':1100, 'loc_f3':2700, 'asp':None, 'b_db':-18},
    'T_AR': {'cl':65, 'burst':3300, 'vb':0.0, 'loc_f2':1100, 'loc_f3':2700, 'asp':'S_AR', 'b_db':-10},
}
