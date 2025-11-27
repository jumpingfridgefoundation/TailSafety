
import numpy as np
import scipy.signal as signal
import scipy.ndimage as ndimage
import sounddevice as sd
import re
import math
import random

from src.config import (
    SAMPLE_RATE, BLOCK_MS, BLOCK_SAMPLES, BIT_DEPTH,
    PHONEMES, DIPHTHONG_MAP, PLOSIVE_DATA
)
from src.g2p import MultiLingualG2P

# Try to import Cython-compiled synthesis functions (if available)
try:
    from src.synthesis import (
        generate_formant_waves, apply_exponential_envelope,
        fast_iir_filter, apply_noise_gate, normalize_audio
    )
    CYTHON_AVAILABLE = True
except ImportError:
    CYTHON_AVAILABLE = False

# Use Numba JIT compilation for performance
try:
    from src.synthesis_numba import (
        generate_formant_waves_jit, apply_exponential_envelope_jit,
        fast_iir_filter_jit, apply_noise_gate_jit, normalize_audio_jit
    )
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

class TailSafetyEngine:
    def __init__(self, voice='default_female', voice_profile=None):
        self.fs = SAMPLE_RATE
        # Require voice profile dict to be passed
        if voice_profile is None:
            raise ValueError("voice_profile dict must be provided")
        self.voice_profile = voice_profile
        self.voice_key = voice_profile.get('name', 'unknown')
        self.base_pitch = self.voice_profile['base_pitch']
        self.g2p = MultiLingualG2P()
        self.reset_filters()
        self.sentence_energy = 1.0
        self.tempo_clock = 0.0
        self.pitch_contour = []  # Track intonation over utterance

    def reset_filters(self):
        self.zi_f = [np.zeros(2) for _ in range(4)]
        self.zi_tilt = np.zeros(1)
        self.phase_acc = 0.0
        self.last_pitch = 125.0
        self.last_f = [500, 1500, 2500, 3500]
        self.sentence_energy = 1.0

    def soft_clip(self, x):
        return np.tanh(x * 0.95)

    def db_to_lin(self, db):
        if db <= -90: return 0.0
        return 10.0 ** (db / 20.0)

    def parse_text(self, text):
        clean_text = re.sub(r'[^\w\s\.,!?\u0400-\u04FF\u0600-\u06FF]', '', text)
        tokens = re.split(r'([.,!?;])', clean_text)
        full_stream = []
        temp_word_buffer = []
        sentence_counter = 0
        
        for t in tokens:
            if not t.strip(): continue
            if t in ['.',',','!','?']:
                if temp_word_buffer:
                    full_stream.extend(temp_word_buffer); full_stream.append(('PAUSE', 45, 0))
                    temp_word_buffer = []
                ms = 200 if t==',' else 450
                full_stream.append(('PAUSE', ms, 0))
                if t in ['.','!','?']:
                    sentence_counter += 1
                    full_stream.append(('BREATH', 600, 0))
            else:
                if temp_word_buffer:
                    full_stream.extend(temp_word_buffer); full_stream.append(('PAUSE', 45, 0))
                    temp_word_buffer = []
                words = t.split()
                for w in words:
                    pron, is_slow = self.g2p.predict(w)
                    for p in pron:
                        s = 0
                        if p[-1].isdigit(): s = int(p[-1]); p = p.rstrip('012')
                        if p == 'HH': p = 'HH'
                        temp_word_buffer.append((p, 0, s, is_slow))
                    temp_word_buffer.append(('WORD_BOUNDARY', 0, 0))
        if temp_word_buffer: full_stream.extend(temp_word_buffer)
        full_stream.append(('END_OF_STREAM', 3000, 0))
        return full_stream

    def generate_tracks(self, stream_segment):
        tracks = {k: [] for k in ['f1','f2','f3','f4','pitch','AV','AF','mix_s','mix_mid','mix_h','burst']}
        
        for i, item in enumerate(stream_segment):
            ph = item[0]
            if ph == 'BREATH' or ph == 'END_OF_STREAM': self.sentence_energy = 1.0 
            
            if ph == 'END_OF_STREAM':
                dur = item[1]; n = int(dur / BLOCK_MS)
                for _ in range(n):
                    tracks['f1'].append(self.last_f[0]); tracks['f2'].append(self.last_f[1])
                    tracks['f3'].append(self.last_f[2]); tracks['f4'].append(self.last_f[3])
                    tracks['pitch'].append(self.last_pitch)
                    tracks['AV'].append(0.0); tracks['AF'].append(0.0)
                    for k in ['mix_s','mix_mid','mix_h','burst']: tracks[k].append(0)
                continue

            if ph == 'WORD_BOUNDARY': continue 
            if ph not in PHONEMES: continue
            
            dur_ov = item[1]
            stress = item[2]
            is_slow_lang = False
            if len(item) > 3: is_slow_lang = item[3]
            
            # Prosody - improved pitch contours and stress
            self.sentence_energy *= 0.97  # Slightly slower decay
            if self.sentence_energy < 0.45: self.sentence_energy = 0.45
            
            # Better stress and intonation
            pitch_offset = (self.sentence_energy * 18.0)
            if stress:
                pitch_offset += 25.0  # Higher rise for stressed syllables
            else:
                pitch_offset -= 8.0   # Lower for unstressed
            pitch_offset += np.random.uniform(-2, 2)  # Reduce jitter
            target_note = self.base_pitch + pitch_offset
            if target_note > self.base_pitch + 55: target_note = self.base_pitch + 55
            if target_note < 75: target_note = 75
            
            self.tempo_clock += 0.1
            tempo_var = math.sin(self.tempo_clock) * 0.12  # Reduce tempo variation
            
            p_data = PHONEMES[ph]
            base_dur = p_data[0]
            
            # Apply duration scaling from voice profile
            duration_scale = self.voice_profile['duration_scale']
            
            if stress: 
                base_dur *= 1.25
            if is_slow_lang: 
                base_dur *= 1.35 
            if ph in ['PAUSE', 'BREATH']: 
                base_dur = dur_ov
            else:
                base_dur *= (1.0 + tempo_var)
                if not stress and self.sentence_energy > 0.8: 
                    base_dur *= 0.92
            
            # Apply voice profile duration scale
            base_dur *= duration_scale

            tgt_f = p_data[1:5]
            tgt_amp = self.db_to_lin(p_data[5])
            
            if ph == 'HH' and i + 1 < len(stream_segment):
                n_ph = stream_segment[i+1][0]
                if n_ph in PHONEMES and PHONEMES[n_ph][6] in [0, 5]: tgt_f = PHONEMES[n_ph][1:5]

            n = max(1, int(base_dur / BLOCK_MS))
            
            # Synthesis Logic
            if p_data[6] == 5: # Glides
                if ph in DIPHTHONG_MAP:
                    s_ph, e_ph = DIPHTHONG_MAP[ph]
                    start_f = PHONEMES[s_ph][1:5]; end_f = PHONEMES[e_ph][1:5]
                elif ph == 'W': 
                    start_f = p_data[1:5]; end_f = start_f
                    if i+1 < len(stream_segment) and stream_segment[i+1][0] in PHONEMES: 
                        end_f = PHONEMES[stream_segment[i+1][0]][1:5]
                else: start_f = tgt_f; end_f = tgt_f

                for f in range(n):
                    k = (1 - np.cos((f/n)*np.pi))/2
                    curr_f = [start_f[x]+(end_f[x]-start_f[x])*k for x in range(4)]
                    tracks['f1'].append(curr_f[0]); tracks['f2'].append(curr_f[1])
                    tracks['f3'].append(curr_f[2]); tracks['f4'].append(curr_f[3])
                    kp = f / n
                    syllable_arc = math.sin(kp * math.pi) * 8.0 
                    curr_p = self.last_pitch + (target_note - self.last_pitch) * kp + syllable_arc
                    tracks['pitch'].append(curr_p)
                    tracks['AV'].append(tgt_amp); tracks['AF'].append(0.0)
                    for k in ['mix_s','mix_mid','mix_h','burst']: tracks[k].append(0)
                self.last_pitch = target_note; self.last_f = end_f

            elif p_data[6] == 2: # Plosives
                key = ph if ph in PLOSIVE_DATA else 'T'
                dat = PLOSIVE_DATA[key]
                for _ in range(int(dat['cl']/BLOCK_MS)):
                    tracks['f1'].append(200); tracks['f2'].append(dat['loc_f2']); tracks['f3'].append(dat['loc_f3']); tracks['f4'].append(3500)
                    tracks['pitch'].append(self.last_pitch); tracks['AV'].append(dat['vb']); tracks['AF'].append(0.0)
                    for k in ['mix_s','mix_mid','mix_h','burst']: tracks[k].append(0)
                tracks['f1'].append(500); tracks['f2'].append(dat['loc_f2']); tracks['f3'].append(dat['loc_f3']); tracks['f4'].append(3500)
                tracks['pitch'].append(self.last_pitch); tracks['AV'].append(dat['vb']); tracks['AF'].append(0.0)
                for k in ['mix_s','mix_mid','mix_h']: tracks[k].append(0)
                tracks['burst'].append(dat['burst'])
                if dat['asp']:
                    asp_dur = 30 if dat['asp'] != 'SH_HARD' else 120
                    for _ in range(int(asp_dur/BLOCK_MS)):
                        tracks['f1'].append(500); tracks['f2'].append(dat['loc_f2']); tracks['f3'].append(dat['loc_f3']); tracks['f4'].append(3500)
                        tracks['pitch'].append(self.last_pitch); tracks['AV'].append(dat['vb']); tracks['AF'].append(0.9)
                        ms, mm, mh = 0,0,0
                        if 'S' in dat['asp']: ms=1
                        elif 'SH' in dat['asp']: mm=1
                        else: mh=1
                        tracks['mix_s'].append(ms); tracks['mix_mid'].append(mm); tracks['mix_h'].append(mh); tracks['burst'].append(0)
                self.last_f = [500, dat['loc_f2'], dat['loc_f3'], 3500]

            else: # Standard
                av, af = (0.0, 0.0)
                if p_data[6] in [0, 6]: av = tgt_amp 
                elif p_data[6] == 1: af = tgt_amp 
                elif p_data[6] == 4: av = tgt_amp*0.5; af = tgt_amp*0.5
                ms, mm, mh = 0,0,0
                if ph in ['S','Z','S_AR']: ms=1
                elif ph in ['SH','ZH']: mm=1
                elif ph in ['HH','KH','H_AR']: mh=1
                if ph in ['F', 'TH']: ms=0; mm=0.5; mh=0.5; af*=0.8
                if ph in ['Z','Z_AR']: av=tgt_amp*0.8; af=tgt_amp*0.7; ms=1.0
                if ph == 'V': av=tgt_amp*0.8; af=tgt_amp*0.5; mh=0.5; ms=0.2
                if ph == 'GH': av=tgt_amp*0.8; af=tgt_amp*0.4; mh=0.8; ms=0.0
                if ph == 'AIN': av=tgt_amp; af=0.0
                if ph in ['KH','H_AR']: mm=0.5; af*=0.6
                
                for f in range(n):
                    tracks['f1'].append(tgt_f[0]); tracks['f2'].append(tgt_f[1])
                    tracks['f3'].append(tgt_f[2]); tracks['f4'].append(tgt_f[3])
                    kp = f / n
                    syllable_arc = math.sin(kp * math.pi) * 5.0
                    curr_p = self.last_pitch + (target_note - self.last_pitch) * kp + syllable_arc
                    tracks['pitch'].append(curr_p)
                    tracks['AV'].append(av); tracks['AF'].append(af)
                    tracks['mix_s'].append(ms); tracks['mix_mid'].append(mm); tracks['mix_h'].append(mh); tracks['burst'].append(0)
                self.last_pitch = target_note; self.last_f = list(tgt_f)

        for k in tracks:
            arr = np.array(tracks[k], dtype=BIT_DEPTH)
            if len(arr) > 0:
                if k == 'pitch': tracks[k] = ndimage.gaussian_filter1d(arr, sigma=4)
                elif k != 'burst': tracks[k] = ndimage.gaussian_filter1d(arr, sigma=2)
                else: tracks[k] = arr
            else: tracks[k] = arr
        return tracks

    def synthesize(self, tracks):
        if len(tracks['pitch']) == 0: return np.zeros(0)
        n = len(tracks['pitch'])
        total = n * BLOCK_SAMPLES
        out = np.zeros(total, dtype=BIT_DEPTH)
        phase = self.phase_acc
        # Klatt-style: minimize noise, maximize formant filtering
        noise_level = self.voice_profile['noise_level'] * 0.5  # Reduce noise for clarity
        raw_noise = np.random.normal(0, noise_level, total)
        BW = [60.0, 90.0, 130.0, 180.0]  # Slightly wider bandwidths for Klatt
        Gains = [1.0, 0.7, 0.5, 0.2]     # More classic Klatt gain ratios

        for b in range(n):
            start, end = b*BLOCK_SAMPLES, (b+1)*BLOCK_SAMPLES
            f_vals = [tracks['f1'][b], tracks['f2'][b], tracks['f3'][b], tracks['f4'][b]]
            pitch = tracks['pitch'][b]
            av, af = tracks['AV'][b], tracks['AF'][b]
            ms, mm, mh = tracks['mix_s'][b], tracks['mix_mid'][b], tracks['mix_h'][b]
            burst = tracks['burst'][b]

            # Klatt-style voicing source: sawtooth
            f0 = pitch
            inc = f0 / self.fs
            src = np.zeros(BLOCK_SAMPLES)
            for i in range(BLOCK_SAMPLES):
                phase += inc
                if phase >= 1.0: phase -= 1.0
                src[i] = 2.0 * (phase - 0.5)

            # Spectral tilt for brightness
            tilt_coeff = 0.92 + (self.voice_profile['brightness'] * 0.05)
            src, self.zi_tilt = signal.lfilter([1.0], [1.0, -tilt_coeff], src, zi=self.zi_tilt)
            src *= av * 0.18

            # Formant scaling
            formant_scale = self.voice_profile['formant_scale']
            scaled_f = [max(50, f_vals[i] / formant_scale) for i in range(4)]

            # Klatt-style formant filters
            y_mix = np.zeros(BLOCK_SAMPLES)
            for i in range(4):
                freq = max(100, min(scaled_f[i], self.fs/2-100))
                bw = BW[i]
                bc, ac = signal.iirpeak(freq, freq/max(50, bw), fs=self.fs)
                y, self.zi_f[i] = signal.lfilter(bc, ac, src, zi=self.zi_f[i])
                y_mix += y * Gains[i]
            out[start:end] = y_mix

            # Fricatives: less noise, more filtered
            if af > 0.01:
                cn = raw_noise[start:end]
                total_n = np.zeros(BLOCK_SAMPLES)
                if ms > 0:
                    b, a = signal.butter(2, [3200, 5800], 'band', fs=self.fs)
                    total_n += signal.lfilter(b, a, cn) * ms * 0.7
                if mm > 0:
                    b, a = signal.butter(2, [1800, 4500], 'band', fs=self.fs)
                    total_n += signal.lfilter(b, a, cn) * mm * 0.7
                if mh > 0:
                    freq_low = max(300, scaled_f[1]-600)
                    freq_high = min(self.fs/2-100, scaled_f[2]+600)
                    if freq_low < freq_high:
                        b, a = signal.butter(2, [freq_low, freq_high], 'band', fs=self.fs)
                        total_n += signal.lfilter(b, a, cn) * mh * 0.7
                out[start:end] += total_n * af

            # Bursts: classic Klatt pop
            if burst > 100:
                pop = np.random.uniform(-1, 1, BLOCK_SAMPLES) * 2.5
                freq_low = max(50, burst-600)
                freq_high = min(self.fs/2-100, burst+600)
                b, a = signal.butter(2, [freq_low, freq_high], 'band', fs=self.fs)
                out[start:end] += self.soft_clip(signal.lfilter(b, a, pop)) * 0.6

        self.phase_acc = phase
        return out

    def speak(self, text):
        print(f" Synth: '{text}'")
        self.reset_filters() 
        full_stream = self.parse_text(text)
        stream = sd.OutputStream(samplerate=self.fs, channels=1, dtype='float32')
        stream.start()
        current_batch = []
        
        for i, item in enumerate(full_stream):
            current_batch.append(item)
            is_mandatory = item[0] in ['PAUSE', 'BREATH', 'END_OF_STREAM']
            is_boundary = (item[0] == 'WORD_BOUNDARY')
            is_buffer_full = len(current_batch) > 15
            
            if is_mandatory or (is_boundary and is_buffer_full):
                tracks = self.generate_tracks(current_batch)
                if len(tracks['pitch']) > 0:
                    wave = self.synthesize(tracks)
                    # Better filtering pipeline
                    b, a = signal.butter(2, 8500, 'low', fs=self.fs)  # Slightly lower cutoff
                    wave = signal.lfilter(b, a, wave)
                    # Gentle additional high-pass to remove DC
                    b, a = signal.butter(1, 20, 'high', fs=self.fs)
                    wave = signal.lfilter(b, a, wave)
                    wave = self.soft_clip(wave * 1.3)  # Slightly higher compression
                    mx = np.max(np.abs(wave))
                    if mx > 0: wave = (wave/mx) * 0.92  # Better normalization
                    stream.write(wave.astype(np.float32))
                current_batch = []
        stream.stop(); stream.close()
