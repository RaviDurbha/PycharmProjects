from IPython.core.display import display
import numpy as np
import matplotlib.pyplot as plt
from util.QRS_util import *
import wfdb
from ecgdetectors import Detectors
from Test_Code.util import read_ecg, EKG_QRS_detect

'''
Normal Parameters(in seconds) (approx)
PR interval     0.12 - 0.2
RR interval     0.6 - 1.0
QT interval     0.35 - 0.43 (less than 1/2 the RR interval)
QRS complex     0.06-0.1
heart rate      60-100bpm (computed)
QRS, RR, QT and PR intervals are computed
'''

FPATH = 'C:\\Users\\RaViDuRbHa\\Desktop\\Courses_Fall_2020\\mit-bih-arrhythmia-database-1.0.0\\100'

# Read signal and annotation file
record = wfdb.rdrecord(FPATH)
sig_len = record.__dict__['p_signal'].shape[0]  # get signal length
fs = record.__dict__['fs']  # get sampling frequency for current record
t = 1 / fs  # convert sample to time (in seconds)

# display(record.__dict__)  # Parameters attributed to signal in annotation file

# define sample size
X = 0
Y = X + (fs * 6)  # Segment Size in Seconds

# save to csv file
ecg_signal = record.__dict__['p_signal'][X:Y, 0]
data = np.asarray(ecg_signal)
data = data * 200  # 200 is ADC gain
np.savetxt('ecg.csv', data, delimiter=',')
print(len(ecg_signal))

# ECG detection
detectors = Detectors(fs)
r_peaks = detectors.engzee_detector(ecg_signal)
fig = plt.figure()
ax1 = fig.add_subplot(211)
plt.plot(ecg_signal)  # plots actual ECG Signal
# plt.plot(r_peaks, ecg_signal[r_peaks], 'r*')  # Plots the peaks on ECG Signal
# ax1.plot(0, 1.5, label='EZee R-Peak')
# ax1.legend(loc='best')
plt.xlabel("Sample #")
plt.ylabel("Amplitude [mV]")
plt.grid(True)

rr_interval = np.diff(r_peaks) * t  # RR interval in number of samples
HR1 = 60 / rr_interval  # instantaneous HR in BPM
BPM1 = min(HR1)
BPM2 = max(HR1)
BPM3 = 60 * 1000 / np.mean(r_peaks)
BPM4 = (min(HR1) + max(HR1)) / 2
ax2 = fig.add_subplot(212)
plt.plot(HR1, 'b-*')
plt.xlabel("Beat Interval")
plt.ylabel("Heart Beat [BPM]")
ax2.plot(0, max(HR1), label='Heart Rate')
ax2.plot(0, max(HR1), label="min_Value: %.1f BPM" % BPM1)
ax2.plot(0, max(HR1), label="Max_Value: %.1f BPM" % BPM2)
ax2.plot(0, max(HR1), label="Mean_Value: %.1f BPM" % BPM3)
ax2.plot(0, max(HR1), label="Avg_Value: %.1f BPM" % BPM4)
ax2.legend(loc='best')
plt.grid(True)

# print(ecg_signal[r_peaks])
# print(np.diff(r_peaks)[0])
# print(np.diff(r_peaks))  # difference between r_peaks
# original_X = np.linspace((r_peaks[0]), (r_peaks[1]))  # Sample values between RR interval
P_onset = r_peaks[0] + 0.7 * np.diff(r_peaks)[0]
P_offset = r_peaks[0] + 0.9 * np.diff(r_peaks)[0]
t = np.linspace(P_onset, P_offset).astype(int)  # Values between 70% to 90% between two consecutive r_peaks

P_peak = ecg_signal.tolist().index(-0.19)

print(P_peak, t[P_peak])


# QRS complex detection
ecg = read_ecg('C:\\Users\\RaViDuRbHa\\PycharmProjects\\Test_Code\\ecg.csv')
R_peaks, S_point, Q_point = EKG_QRS_detect(ecg, fs, True, True)
QRS_Interval = np.subtract(S_point / fs, Q_point / fs)

#  Min, Max Interval of Feature Extraction
min_rr_value = min(rr_interval) * 1000
max_rr_value = max(rr_interval) * 1000
print("RR_Interval(min): %.3f ms" % min_rr_value)
print("RR_Interval(max): %.3f ms" % max_rr_value)

min_qrs_value = min(QRS_Interval) * 1000
max_qrs_value = max(QRS_Interval) * 1000
print("QRS_Interval(min): %.3f ms" % min_qrs_value)
print("QRS_Interval(max): %.3f ms" % max_qrs_value)

# Test Conditions
if (rr_interval >= 0.6).all() & (rr_interval <= 1.0).all() & (QRS_Interval >= 0.06).all() & (QRS_Interval <= 0.1).all():
    print('ECG is Normal')
else:
    print('ECG is Abnormal')

# Metrics
rmssd = np.sqrt(np.mean(np.square(np.diff(rr_interval))))  # RMSSD (square root of the mean square of the differences)
sdnn = np.std(rr_interval)  # SDNN - Compute the standard deviation along the specified axis
print("RMSSD: %.3f ms" % rmssd)
print("SDNN: %.3f ms" % sdnn)

plt.show()
