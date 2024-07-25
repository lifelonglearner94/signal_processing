import numpy as np
from scipy import signal

def simple_ma_detrending(input_signal):

    # Ensure window_size is odd
    if window_size % 2 == 0:
        window_size += 1

    # Create the moving average filter
    window = np.ones(window_size) / window_size

    # Compute the moving average
    # Use 'same' mode to keep the output size same as input
    # and 'valid' mode for the convolution to handle edge effects
    ma = np.convolve(input_signal, window, mode='same')

    # Subtract the moving average from the original signal
    detrended_ecg = input_signal - ma

    return detrended_ecg


def find_frequency(input_signal, threshold):
    # Find peaks that exceed a threshold value
    peak_times, _ = signal.find_peaks(input_signal, height=threshold)

    # Calculate intervals between the peaks
    peak_intervals = np.diff(peak_times)

    median_difference_between_peaks = np.median(peak_intervals)

    # median_difference_between_peaks * average rest-heartbeat per minute / seconds of a minute = sampling frequency in Hz
    estimated_freq = median_difference_between_peaks * 70 / 60

    return estimated_freq



def highpass_process_signal(input_signal, sampling_rate=200, detrend_cutoff=0.3, smooth_cutoff=2.5):

    nyquist_freq = 0.5 * sampling_rate

    # Design high-pass filter for detrending
    high_normal_cutoff = detrend_cutoff / nyquist_freq
    b_high, a_high = signal.butter(4, high_normal_cutoff, btype='high', analog=False)

    # Design low-pass filter for smoothing
    low_normal_cutoff = smooth_cutoff / nyquist_freq
    b_low, a_low = signal.butter(4, low_normal_cutoff, btype='low', analog=False)

    # Apply high-pass filter (detrending)
    detrended_ecg = signal.filtfilt(b_high, a_high, input_signal)

    # Apply low-pass filter (smoothing)
    processed_ecg = signal.filtfilt(b_low, a_low, detrended_ecg)

    return processed_ecg
