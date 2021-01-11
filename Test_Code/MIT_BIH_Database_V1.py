"""
P-QRS-T wave peaks, onset and offset detection
@author : ravi.durbha@tufts.edu
"""

from IPython.core.display import display
import numpy as np
import matplotlib.pyplot as plt
from util.QRS_util import *
import wfdb
from Test_Code.util import read_ecg, EKG_QRS_detect

FPATH = 'C:\\Users\\RaViDuRbHa\\Desktop\\Wireless_Medical_Devices\\mit-bih-arrhythmia-database-1.0.0\\100'

# Read signal and annotation file
record = wfdb.rdrecord(FPATH)
sig_len = record.__dict__['p_signal'].shape[0]  # get signal length
fs = record.__dict__['fs']  # get sampling frequency for current record
ADC_Gain = record.__dict__['adc_gain']  # get adc_gain for current record
t = 1 / fs  # convert sampling frequency to time (in seconds)

# display(record.__dict__)  # Parameters attributed to signal in annotation file

# # define sample size
X = 0
Y = X + (fs * 6)  # Segment Size in Seconds

# # save to csv file
ecg_signal = record.__dict__['p_signal'][X:Y, 0]
data = np.asarray(ecg_signal).round(decimals=3)
data = data * ADC_Gain[0]
np.savetxt('ecg.csv', data, delimiter=',')
size = len(ecg_signal)
# print("length of ecg_signal: %.0f " % size)

# # QRS Complex Detection
ecg = read_ecg('C:\\Users\\RaViDuRbHa\\PycharmProjects\\Test_Code\\ecg.csv')
R_peaks, S_point, Q_point = EKG_QRS_detect(ecg, fs, True, True)
QRS_Interval = np.subtract(S_point / fs, Q_point / fs)  # QRS Complex Detection

# print(R_peaks)  # R-peaks Sample No's

fig = plt.figure()
ax1 = fig.add_subplot(211)
plt.plot(ecg_signal)  # plots actual ECG Signal
# plt.plot(R_peaks, ecg_signal[R_peaks], 'r*')  # Plots the peaks on ECG Signal
plt.xlabel("Sample #")
plt.ylabel("Amplitude [mV]")
plt.grid(True)

rr_interval = np.diff(R_peaks) * t  # RR interval in number of samples
HR1 = 60 / rr_interval  # instantaneous HR in BPM
BPM1 = min(HR1)
BPM2 = max(HR1)
# BPM3 = (min(HR1) + max(HR1)) / 2
# BPM4 = 60 * 1000 / np.mean(R_peaks)
ax2 = fig.add_subplot(212)
plt.plot(HR1, 'b-*')
plt.xlabel("Beat Interval")
plt.ylabel("Heart Beat [BPM]")
ax2.plot(0, max(HR1), label='Heart Rate')
ax2.plot(0, max(HR1), label="min_Value: %.1f BPM" % BPM1)
ax2.plot(0, max(HR1), label="Max_Value: %.1f BPM" % BPM2)
# ax2.plot(0, max(HR1), label="Avg_Value: %.1f BPM" % BPM3)
# ax2.plot(0, max(HR1), label="Mean_Value: %.1f BPM" % BPM4)
ax2.legend(loc='best')
plt.grid(True)

# # QRS_onset Detection
F2 = []
for r in range(0, len(R_peaks)):
    F1 = []  # Clear vectors from previous cardiac cycle and update with new vectors
    qrs_on = Q_point[r] - (0.042 * fs)
    onset_val = np.arange(qrs_on, Q_point[r] + 1, 1, dtype=int)  # Calculate QRS onset Range
    for x in range(1, len(onset_val)):
        slope = (ecg_signal[onset_val][x] - ecg_signal[onset_val][x - 1]) / (onset_val[x] - onset_val[x - 1])
        if slope > 0:
            F1.append((onset_val[x]))
    F2.append(max(F1))
QRSon_count = np.array(F2)
# print(QRSon_count, ecg_signal[QRSon_count])

# # QRS_offset Detection
G2 = []
for s in range(0, len(R_peaks)):
    G1 = []  # Clear vectors from previous cardiac cycle and update with new vectors
    qrs_off = S_point[s] + (0.042 * fs)
    offset_val = np.arange(S_point[s], qrs_off, 1, dtype=int)  # Calculate QRS offset Range
    for i in range(1, len(offset_val)):
        slope = (ecg_signal[offset_val][i] - ecg_signal[offset_val][i - 1]) / (offset_val[i] - offset_val[i - 1])
        if slope < 0:
            G1.append(offset_val[i] - 1)
    G2.append(min(G1))
QRS_off_count = np.array(G2)
# print(QRS_off_count, ecg_signal[QRS_off_count])

# # P_onset P_offset and P-peaks Detection
k1 = []
k11 = []
k2 = []
k22 = []
k3 = []
k33 = []
for r in range(0, len(R_peaks)):
    P_LL1 = R_peaks[r] - 100
    P_UL1 = R_peaks[r] - 50
    p1 = np.arange(P_LL1, P_UL1 + 1, dtype=int)  # Calculate P-wave onset and offset range (X-axis)
    p1_val = np.argmax(ecg_signal[p1])  # Calculate P-wave amplitudes (Y-axis)
    P1_peak = p1[p1_val]  # P-peak Value
    k1.append(P1_peak)
    k11.append(max(ecg_signal[p1]))
    k2.append(P_LL1)
    k22.append(ecg_signal[P_LL1])
    k3.append(P_UL1)
    k33.append(ecg_signal[P_UL1])
Pon_val = np.array(k2)
Pon_amp = np.array(k22)
Pwav_val = np.array(k1)
Pwav_amp = np.array(k11)
Poff_val = np.array(k3)
Poff_amp = np.array(k33)
# print(Pon_val, Pwav_val, Poff_val)

# # T-wave Detection
S1 = []
S11 = []
S2 = []
S22 = []
S3 = []
S33 = []
for t in range(0, len(R_peaks)):
    T_LL = R_peaks[t] + 25
    T_UL = R_peaks[t] + 100
    if T_LL >= size:
        continue
    elif T_UL > size:
        T_UL = size - 1  # Reshaping T-wave offset vector if greater than size of ecg_signal
    T = np.arange(T_LL, T_UL + 1, dtype=int)  # Calculate T-wave onset and offset range (X-axis)
    T_val = np.argmax(ecg_signal[T])  # Calculate T-wave amplitudes (Y-axis)
    T_peak = T[T_val]  # T_peak Value
    S1.append(T_peak)
    S11.append(max(ecg_signal[T]))
    S2.append(T_LL)
    S22.append(ecg_signal[T_LL])
    S3.append(T_UL)
    S33.append(ecg_signal[T_UL])
Twav_val = np.array(S1)
Twav_amp = np.array(S11)
Ton_val = np.array(S2)
Ton_amp = np.array(S22)
Toff_val = np.array(S3)
Toff_amp = np.array(S33)
# print(Ton_val, Twav_val, Toff_val)

# # Wave_Intervals/Feature Extraction # #
'''
Ref: emedicine.medscape.com/article/2172196-overview
0.6 >= rr_interval <= 1.0
60 >= HR1 <= 100
0.12 >= PR_interval <= 0.2
0.05 >= PR_Segment <= 0.12
0.06 >= QRS_complex <= 0.12
0.08 >= ST_segment <= 0.12y
0.35 >= QT_interval <= 0.43
'''
RR_Interval = rr_interval  # RR Interval
PR_interval = np.subtract(QRSon_count / fs, Pon_val / fs)  # PR Interval
PR_Segment = np.subtract(QRSon_count / fs, Poff_val / fs)  # PR Segment
QRS_complex = np.subtract(QRS_off_count / fs, QRSon_count / fs)  # QRS Complex
ST_segment = np.subtract(Ton_val / fs, QRS_off_count / fs)  # ST Segment
QT_interval = np.subtract(Toff_amp / fs, QRSon_count / fs)  # QT Interval
Pwav_duration = np.subtract(Poff_val / fs, Pon_val / fs)  # P-Wave Duration
P_Wave_amplitude = Pwav_amp  # P-wave Amplitude
Twav_duration = np.subtract(Toff_val / fs, Ton_val / fs)  # T-Wave Duration
T_Wave_amplitude = Twav_amp  # T-wave Amplitude

# # Test Conditions
if (rr_interval >= 0.6).all() & (rr_interval <= 1.2).all() & (QRS_complex >= 0.06).all() & (QRS_complex <= 0.12).all():
    print('Normal Sinus Rhythm')
else:
    print('Abnormal')

# # Metrics
rmssd = np.sqrt(np.mean(np.square(np.diff(rr_interval))))  # RMSSD (square root of the mean square of the differences)
sdnn = np.std(rr_interval)  # SDNN - Compute the standard deviation along the specified axis
print("RMSSD: %.3f ms" % rmssd)
print("SDNN: %.3f ms" % sdnn)

plt.show()
