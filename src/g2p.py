import re
try:
    from g2p_en import G2p as G2P_EN
    G2P_ENGLISH = G2P_EN()
except ImportError:
    print("Installing g2p-en...")
    import subprocess
    subprocess.run(['pip', 'install', 'g2p-en'], check=True)
    from g2p_en import G2p as G2P_EN
    G2P_ENGLISH = G2P_EN()


class MultiLingualG2P:
    def __init__(self):
        """Initialize multilingual G2P using public libraries"""
        self.g2p_en = G2P_ENGLISH
        
        # Russian character to phoneme mapping
        self.ru_map = {
            'А': ['AA'], 'Б': ['B'], 'В': ['V'], 'Г': ['G'], 'Д': ['D'], 'Е': ['IY', 'EH'],
            'Ё': ['IY', 'AO'], 'Ж': ['ZH'], 'З': ['Z'], 'И': ['IY'], 'Й': ['Y'], 'К': ['K'],
            'Л': ['L'], 'М': ['M'], 'Н': ['N'], 'О': ['AO'], 'П': ['P'], 'Р': ['RR'],
            'С': ['S'], 'Т': ['T'], 'У': ['UW'], 'Ф': ['F'], 'Х': ['KH'], 'Ц': ['T', 'S'],
            'Ч': ['CH'], 'Ш': ['SH'], 'Щ': ['SH', 'CH'], 'Ъ': ['PAUSE'], 'Ы': ['IH'],
            'Ь': [], 'Э': ['EH'], 'Ю': ['Y', 'UW'], 'Я': ['Y', 'AA']
        }
        
        # Arabic character to phoneme mapping
        self.ar_cons = {
            'ا': 'AA', 'ب': 'B', 'ت': 'T', 'ث': 'TH', 'ج': 'JH', 'ح': 'H_AR',
            'خ': 'KH', 'د': 'D', 'ذ': 'DH', 'ر': 'RR', 'ز': 'Z', 'س': 'S',
            'ش': 'SH', 'ص': 'S_AR', 'ض': 'D_AR', 'ط': 'T_AR', 'ظ': 'Z_AR',
            'ع': 'AIN', 'غ': 'GH', 'ف': 'F', 'ق': 'Q', 'ك': 'K', 'ل': 'L',
            'م': 'M', 'ن': 'N', 'ه': 'HH', 'و': 'UW', 'ي': 'IY', 'ة': 'T',
            'ء': 'Q', 'ؤ': 'Q', 'ئ': 'Q', 'ى': 'AA'
        }

    def detect_script(self, text):
        """Detect script type: English, Russian, or Arabic"""
        if re.search(r'[\u0400-\u04FF]', text):
            return 'RU'
        if re.search(r'[\u0600-\u06FF]', text):
            return 'AR'
        return 'EN'

    def predict_english(self, word):
        """Convert English text to phonemes using g2p-en"""
        try:
            phonemes = self.g2p_en(word)
            # Filter out stress markers and return clean phonemes
            return [p for p in phonemes if p.strip() and p not in ['', ' ']]
        except Exception as e:
            print(f"Error in English G2P: {e}")
            return ['AH']

    def predict_russian(self, word):
        """Convert Russian text to phonemes"""
        phonemes = []
        word = word.upper()
        for char in word:
            if char in self.ru_map:
                phonemes.extend(self.ru_map[char])
        return phonemes

    def predict_arabic(self, word):
        """Convert Arabic text to phonemes"""
        phonemes = []
        for char in word:
            if char in self.ar_cons:
                phonemes.append(self.ar_cons[char])
        return phonemes if phonemes else ['AH']

    def predict(self, word):
        """Predict phonemes for word in any language"""
        script = self.detect_script(word)
        if script == 'RU':
            return (self.predict_russian(word), False)
        elif script == 'AR':
            return (self.predict_arabic(word), True)
        else:
            return (self.predict_english(word), False)
