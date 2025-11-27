"""
Numba JIT-compiled synthesis functions
Compiles to machine code for maximum performance without C compiler
"""

import numpy as np
from numba import jit, prange
import math

@jit(nopython=True, fastmath=True)
def generate_formant_waves_jit(
    time_array,
    num_samples,
    base_freq,
    formant_freq,
    bandwidth,
    amplitude,
    envelope_value
):
    """
    Generate formant resonance using second-order resonator
    JIT-compiled to machine code
    """
    wave = np.zeros(num_samples, dtype=np.float64)
    fs = 48000.0
    
    # Resonator coefficients
    B = bandwidth / fs
    A1 = -2.0 * math.cos(2.0 * math.pi * formant_freq / fs)
    A2 = 1.0 - 2.0 * B + (B * B)
    A1 = A1 * A2
    
    # Normalization
    b0 = B * B
    
    # Generate resonance response
    y0, y1, y2, x0 = 0.0, 0.0, 0.0, 0.0
    
    for i in range(num_samples):
        t = time_array[i]
        # Excitation signal
        x0 = math.sin(2.0 * math.pi * base_freq * t)
        
        # IIR filter (second-order)
        y0 = b0 * x0 + A1 * y1 + A2 * y2
        y2 = y1
        y1 = y0
        
        # Apply envelope and amplitude
        wave[i] = y0 * amplitude * envelope_value
    
    return wave


@jit(nopython=True, fastmath=True)
def apply_exponential_envelope_jit(
    signal,
    attack_samples,
    sustain_samples,
    release_samples,
    sustain_level
):
    """
    Apply ADSR envelope to signal with exponential curves
    JIT-compiled for real-time performance
    """
    num_samples = len(signal)
    envelope = np.ones(num_samples, dtype=np.float64)
    total_envelope_samples = attack_samples + sustain_samples + release_samples
    
    if total_envelope_samples == 0:
        return signal * envelope
    
    # Clamp to signal length
    if total_envelope_samples > num_samples:
        total_envelope_samples = num_samples
    
    # Attack phase (0 to 1)
    if attack_samples > 0:
        for i in range(attack_samples):
            if i < num_samples:
                t_norm = float(i) / float(attack_samples)
                # Exponential attack
                envelope[i] = 1.0 - math.exp(-5.0 * t_norm)
    
    # Sustain phase (constant)
    for i in range(attack_samples, attack_samples + sustain_samples):
        if i < num_samples:
            envelope[i] = sustain_level
    
    # Release phase
    for i in range(attack_samples + sustain_samples, total_envelope_samples):
        if i < num_samples:
            t_norm = float(i - attack_samples - sustain_samples) / float(release_samples)
            # Exponential release
            envelope[i] = sustain_level * math.exp(-5.0 * t_norm)
    
    # Everything after envelope is zero
    for i in range(total_envelope_samples, num_samples):
        envelope[i] = 0.0
    
    return signal * envelope


@jit(nopython=True, fastmath=True)
def fast_iir_filter_jit(
    signal,
    b_coeffs,
    a_coeffs
):
    """
    Fast IIR filter implementation (JIT compiled)
    Direct Form II
    """
    num_samples = len(signal)
    num_b = len(b_coeffs)
    num_a = len(a_coeffs)
    output = np.zeros(num_samples, dtype=np.float64)
    state_len = max(num_b, num_a) - 1
    state = np.zeros(state_len, dtype=np.float64)
    
    for i in range(num_samples):
        x = signal[i]
        
        # FIR part
        y = b_coeffs[0] * x
        for j in range(1, num_b):
            if j - 1 < state_len:
                y += b_coeffs[j] * state[j - 1]
        
        # IIR part
        if num_a > 1:
            y = y / a_coeffs[0]
            for j in range(1, min(num_a, state_len + 1)):
                y -= a_coeffs[j] * state[j - 1] / a_coeffs[0]
        
        # Shift state
        for j in range(state_len - 1, 0, -1):
            state[j] = state[j - 1]
        
        if state_len > 0:
            state[0] = y
        
        output[i] = y
    
    return output


@jit(nopython=True, fastmath=True)
def apply_noise_gate_jit(
    signal,
    threshold,
    attack_ms,
    release_ms,
    sample_rate
):
    """
    Fast noise gate implementation (JIT compiled)
    """
    num_samples = len(signal)
    attack_samples = int(attack_ms * sample_rate / 1000.0)
    release_samples = int(release_ms * sample_rate / 1000.0)
    output = signal.copy()
    
    gate_open = 0
    window_size = 256
    
    for i in range(0, num_samples - window_size, window_size // 2):
        # Calculate RMS of window
        rms = 0.0
        for j in range(i, min(i + window_size, num_samples)):
            rms += signal[j] * signal[j]
        rms = math.sqrt(rms / float(window_size))
        
        # Gate logic
        if rms > threshold:
            gate_open = 1
        elif gate_open and i > release_samples:
            gate_open = 0
        
        # Apply gate
        factor = float(gate_open)
        for j in range(i, min(i + window_size // 2, num_samples)):
            output[j] = signal[j] * factor
    
    return output


@jit(nopython=True, fastmath=True)
def normalize_audio_jit(
    signal,
    target_level
):
    """
    Fast audio normalization (JIT compiled)
    """
    num_samples = len(signal)
    output = np.zeros(num_samples, dtype=np.float64)
    
    # Find peak
    peak = 0.0
    for i in range(num_samples):
        abs_val = abs(signal[i])
        if abs_val > peak:
            peak = abs_val
    
    # Normalize
    if peak > 0.0:
        factor = target_level / peak
        for i in range(num_samples):
            output[i] = signal[i] * factor
    else:
        output = signal.copy()
    
    return output
