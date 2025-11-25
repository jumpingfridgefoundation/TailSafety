
import re
import nltk

try:
    from nltk.corpus import cmudict
    CMU_DICT = cmudict.dict()
except LookupError:
    print("Downloading CMU Dictionary...")
    nltk.download('cmudict', quiet=True)
    from nltk.corpus import cmudict
    CMU_DICT = cmudict.dict()

# Try to import Mishkal for Arabic
ARABIC_TASHKEEL = None
try:
    import mishkal.tashkeel
    ARABIC_TASHKEEL = mishkal.tashkeel.TashkeelClass()
except ImportError:
    pass

class MultiLingualG2P:
    def __init__(self):
        self.en_rules = [
            ('TION', ['SH', 'AH', 'N']), ('ING', ['IH', 'NG']), ('OUS', ['AH', 'S']),
            ('IGHT', ['AY', 'T']), ('OUGH', ['OW']), ('EE', ['IY']), ('EA', ['IY']), 
            ('OO', ['UW']), ('AI', ['EY']), ('AY', ['EY']), ('OA', ['OW']), ('OW', ['OW']), 
            ('OU', ['AW']), ('AU', ['AO']), ('AR', ['AA', 'R']), ('SH', ['SH']), 
            ('CH', ['CH']), ('TH', ['TH']), ('PH', ['F']), ('WH', ['W'])
        ]
        self.ru_map = {
            'А':['AA'], 'Б':['B'], 'В':['V'], 'Г':['G'], 'Д':['D'], 'Е':['IY','EH'], 
            'Ё':['IY','AO'], 'Ж':['ZH'], 'З':['Z'], 'И':['IY'], 'Й':['Y'], 'К':['K'], 
            'Л':['L'], 'М':['M'], 'Н':['N'], 'О':['AO'], 'П':['P'], 'Р':['RR'], 
            'С':['S'], 'Т':['T'], 'У':['UW'], 'Ф':['F'], 'Х':['KH'], 'Ц':['T','S'], 
            'Ч':['CH'], 'Ш':['SH'], 'Щ':['SH','CH'], 'Ъ':['PAUSE'], 'Ы':['IH'], 
            'Ь':[], 'Э':['EH'], 'Ю':['Y','UW'], 'Я':['Y','AA']
        }
        self.ar_cons = {
            'ا': 'AA', 'ب': 'B', 'ت': 'T', 'ث': 'TH', 'ج': 'JH', 'ح': 'H_AR',
            'خ': 'KH', 'د': 'D', 'ذ': 'DH', 'ر': 'RR', 'ز': 'Z', 'س': 'S',
            'ش': 'SH', 'ص': 'S_AR', 'ض': 'D_AR', 'ط': 'T_AR', 'ظ': 'Z_AR',
            'ع': 'AIN', 'غ': 'GH', 'ف': 'F', 'ق': 'Q', 'ك': 'K', 'ل': 'L',
            'م': 'M', 'ن': 'N', 'ه': 'HH', 'و': 'UW', 'ي': 'IY', 'ة': 'T',
            'ء': 'Q', 'ؤ': 'Q', 'ئ': 'Q', 'ى': 'AA'
        }
        self.sun_letters = set(['ت','ث','د','ذ','ر','ز','س','ش','ص','ض','ط','ظ','ل','ن'])
        self.emphatic_letters = set(['ص','ض','ط','ظ','ق','غ','خ'])
        self.FATHA = '\u064E'; self.DAMMA = '\u064F'; self.KASRA = '\u0650'
        self.FATHATAN = '\u064B'; self.DAMMATAN = '\u064C'; self.KASRATAN = '\u064D'
        self.SHADDA = '\u0651'; self.ALEF_MADDA = '\u0622'

    def detect_script(self, text):
        if re.search(r'[\u0400-\u04FF]', text): return 'RU'
        if re.search(r'[\u0600-\u06FF]', text): return 'AR'
        return 'EN'

    def predict_russian(self, word):
        phonemes = []; word = word.upper()
        for char in word:
            if char in self.ru_map: phonemes.extend(self.ru_map[char])
        return phonemes

    def predict_arabic(self, word):
        if ARABIC_TASHKEEL:
            try: word = ARABIC_TASHKEEL.tashkeel(word)
            except: pass
        else:
            vowelized = ""
            for i, c in enumerate(word):
                vowelized += c
                if c in self.ar_cons and i+1 < len(word):
                    next_c = word[i+1]
                    if next_c in self.ar_cons and next_c not in ['ا','و','ي']:
                        vowelized += self.FATHA
            word = vowelized

        phonemes = []; i = 0
        while i < len(word):
            c = word[i]
            if c == 'ا' and i+2 < len(word) and word[i+1] == 'ل' and word[i+2] in self.sun_letters:
                phonemes.append('AE'); i += 2; c = word[i]
                pass 
            if c == self.ALEF_MADDA:
                phonemes.extend(['Q', 'AA']); i += 1; continue
            if c not in self.ar_cons:
                i += 1; continue
            p = self.ar_cons[c]
            is_shadda = False
            if i+1 < len(word) and word[i+1] == self.SHADDA:
                is_shadda = True; i += 1
            phonemes.append(p)
            if is_shadda: phonemes.append(p)
            
            vowel = None
            if i+1 < len(word):
                nxt = word[i+1]
                is_emphatic = c in self.emphatic_letters or c == 'ر'
                if nxt == self.FATHA: vowel = 'AA' if is_emphatic else 'AE'; i+=1
                elif nxt == self.DAMMA: vowel = 'UH'; i+=1
                elif nxt == self.KASRA: vowel = 'IH'; i+=1
                elif nxt == self.FATHATAN: 
                    vowel = 'AA' if is_emphatic else 'AE'; phonemes.append(vowel); phonemes.append('N'); vowel=None; i+=1
                elif nxt == self.DAMMATAN: phonemes.append('UH'); phonemes.append('N'); i+=1
                elif nxt == self.KASRATAN: phonemes.append('IH'); phonemes.append('N'); i+=1
                elif nxt == 'ا': vowel = 'AA'
            if vowel: phonemes.append(vowel)
            i += 1
        return phonemes

    def predict_english(self, word):
        word = word.upper(); phonemes = []; i = 0
        while i < len(word):
            match = False
            for pat, phones in self.en_rules:
                if word[i:].startswith(pat):
                    phonemes.extend(phones); i += len(pat); match = True; break
            if not match: 
                c = word[i]; i += 1
                if c in ['A','E','I','O','U']: phonemes.append('AH')
                elif c in ['S','T','R','L']: phonemes.append(c)
                else: phonemes.append('T')
        return phonemes

    def predict(self, word):
        script = self.detect_script(word)
        if script == 'RU': return (self.predict_russian(word), False)
        elif script == 'AR': return (self.predict_arabic(word), True)
        else:
            if word.lower() in CMU_DICT: 
                raw = CMU_DICT[word.lower()][0]
                clean = [p.rstrip('012') for p in raw]
                return (clean, False)
            return (self.predict_english(word), False)
