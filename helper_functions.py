import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

def simple_ma_detrending(input_signal, window_size=10):

    # Create the moving average filter
    window = np.ones(window_size) / window_size

    # Compute the moving average
    ma = np.convolve(input_signal, window, mode='same')

    # Subtract the moving average from the original signal
    detrended_ecg = input_signal - ma

    return detrended_ecg


def plot_three_signals(signal1, signal2, signal3):
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].plot(signal1)
    axs[1].plot(signal2)
    axs[2].plot(signal3)
    plt.show()

def find_frequency(input_signal, threshold):
    # Find peaks that exceed a threshold value
    peak_times, _ = signal.find_peaks(input_signal, height=threshold)

    # Calculate intervals between the peaks
    peak_intervals = np.diff(peak_times)

    median_difference_between_peaks = np.median(peak_intervals)

    # Median_difference_between_peaks * average rest-heartbeat per minute / seconds of a minute = sampling frequency in Hz
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


def find_pulse_locations(input_signal, threshold):
    # Find peaks that exceed a threshold value
    peak_times, _ = signal.find_peaks(input_signal, height=threshold)

    return peak_times


def plot_signals_with_peaks(signal1, signal2, signal3, peaks1, peaks2, peaks3, xlim, ylim):

    signals = [signal1, signal2, signal3]
    peaks = [peaks1, peaks2, peaks3]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for i, ax in enumerate(axes):
        # Line plot of the signal
        ax.plot(signals[i])

        # Scatter plot of the peaks
        y_values = [signals[i][j] for j in peaks[i]]
        ax.scatter(peaks[i], y_values, color='red')

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

    plt.show()


def calculate_zscore(array: np.ndarray):
    # (X−μ)​ / σ
    zscore = (array - np.mean(array)) / np.std(array)
    return zscore

def anomaly_detection(peaks):
    peak_diff = np.diff(peaks)
    peak_diff_z = calculate_zscore(peak_diff)

    # The data is different, so we need to compute a reasonable threshold
    Q1 = np.percentile(np.abs(peak_diff_z), 25)
    Q3 = np.percentile(np.abs(peak_diff_z), 75)
    IQR = Q3 - Q1
    # After some testing, this was a good formular to find some anomalies in every signal
    anomaly_threshold = Q3 + 0.5 * IQR

    anomalies = np.where((np.abs(peak_diff_z) > anomaly_threshold))

    return anomalies

def synchronize_signals(input_signal_ecg, input_signal_ppg, ecg_pulse_locations, ppg_pulse_locations):

    ecg_diff = np.median(np.diff(ecg_pulse_locations))
    ppg_diff = np.median(np.diff(ppg_pulse_locations))

    scaling_factor = ecg_diff / ppg_diff

    resampled_ppg = signal.resample(input_signal_ppg, int(len(input_signal_ppg) * scaling_factor))

    cross_corr = signal.correlate(input_signal_ecg, resampled_ppg, mode='full')

    max_corr = np.max(cross_corr)

    lag = np.argmax(cross_corr) - (len(resampled_ppg) - 1)


    if lag > 0:
        aligned_resampled_ppg = np.pad(resampled_ppg, (lag, 0), mode='constant')[:len(input_signal_ecg)]
    else:
        aligned_resampled_ppg = np.pad(resampled_ppg, (0, -lag), mode='constant')[:len(input_signal_ecg)]


    # output_ppg = aligned_resampled_ppg[lag:]
    # output_ecg = input_signal_ecg[lag:]

    # Plotting the data
    #plt.plot(resampled_ppg, label='Resampled PPG')
    plt.plot(aligned_resampled_ppg, label='Aligned Resampled PPG')
    plt.plot(calculate_zscore(input_signal_ecg), label='Z-scored ECG')

    plt.xlim(80000, 84000)
    plt.ylim(-10, 10)

    plt.legend()

    plt.show()

    return aligned_resampled_ppg

def align_and_trim_signals(signal1, signal2, threshold=1e-6):
    # Find start and end indices of non-padding for both signals
    start1 = np.argmax(np.abs(signal1) > threshold)
    end1 = len(signal1) - np.argmax(np.abs(signal1[::-1]) > threshold)

    start2 = np.argmax(np.abs(signal2) > threshold)
    end2 = len(signal2) - np.argmax(np.abs(signal2[::-1]) > threshold)

    # Determine common start and end to maintain alignment
    common_start = max(start1, start2)
    common_end = min(end1, end2)

    # Trim both signals
    trimmed_signal1 = signal1[common_start:common_end]
    trimmed_signal2 = signal2[common_start:common_end]

    return trimmed_signal1, trimmed_signal2
