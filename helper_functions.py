import numpy as np
from scipy import signal, stats
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


def iterativly_find_best_synch(input_signal_ecg, input_signal_ppg, ecg_pulse_locations, ppg_pulse_locations):

    # Using the information that ECG has started before PPG and stopped afterwards I do the following calulation:
    number_of_pulses_ecg = len(ecg_pulse_locations)
    number_of_pulses_ppg = len(ppg_pulse_locations)

    diff_number_pulses = number_of_pulses_ecg - number_of_pulses_ppg

    ecg_median_timesteps_between_pulses = np.median(np.diff(ecg_pulse_locations))

    timesteps_ecg_is_longer_than_ppg = ecg_median_timesteps_between_pulses * diff_number_pulses

    final_corr_coefs = []
    ecg_signals = []
    ppg_signals = []

    for iterator in range(2000):

        half_timesteps_ecg_is_longer_than_ppg = int((timesteps_ecg_is_longer_than_ppg / 2) + iterator)

        # Cut ECG signal at start and end
        cutted_ecg_signal = input_signal_ecg[half_timesteps_ecg_is_longer_than_ppg:len(input_signal_ecg)-half_timesteps_ecg_is_longer_than_ppg]

        # Bring PPG to the same length as ECG
        resampled_ppg = signal.resample(input_signal_ppg, len(cutted_ecg_signal))

        # Find the number of timeshifts needed for the signals to match best
        cross_corr = signal.correlate(cutted_ecg_signal, resampled_ppg, mode='full')

        # Find timeshift where the signal matches best (result of cross-corr is around double the length of original signal)
        lag = np.argmax(cross_corr) - (len(resampled_ppg) - 1)

        # if lag bigger 0 a shift to the right is needed, else a shift to the left
        if lag > 0:
            aligned_resampled_ppg = np.pad(resampled_ppg, (lag, 0), mode='constant')[:len(cutted_ecg_signal)]
        else:
            aligned_resampled_ppg = np.pad(resampled_ppg, (0, -lag), mode='constant')[:len(cutted_ecg_signal)]

        # Evaluate to find the best matching signals
        final_corr_coef = stats.pearsonr(aligned_resampled_ppg, cutted_ecg_signal)[0]
        final_corr_coefs.append(final_corr_coef)

        ecg_signals.append(cutted_ecg_signal)
        ppg_signals.append(aligned_resampled_ppg)

    # Get the index for the highest correlation
    max_index = final_corr_coefs.index(max(final_corr_coefs))

    best_ecg = ecg_signals[max_index]
    best_ppg = ppg_signals[max_index]

    return best_ecg, best_ppg


def plot_all_signals(first_ecg, first_ppg, second_ecg, second_ppg, third_ecg, third_ppg):
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    axs[0].plot(calculate_zscore(first_ecg), label='Z-Scored ECG')
    axs[0].plot(first_ppg, label='PPG')
    axs[0].set_title('First Signals')
    axs[0].legend()

    axs[1].plot(calculate_zscore(second_ecg), label='Z-Scored ECG')
    axs[1].plot(calculate_zscore(second_ppg), label='Z-Scored PPG')
    axs[1].set_title('Second Signals')
    axs[1].legend()

    axs[2].plot(calculate_zscore(third_ecg), label='Z-Scored ECG')
    axs[2].plot(calculate_zscore(third_ppg), label='Z-Scored PPG')
    axs[2].set_title('Third Signals')
    axs[2].legend()

    fig.suptitle('ECG and PPG Signals Comparison', fontsize=16)

    plt.tight_layout()
    plt.show()
