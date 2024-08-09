import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd
import tkinter as tk
from tkinter import ttk, messagebox
from itertools import chain
import mne
import os

# costum libraries
from eeg_decoder import Decoder, findall_datadate_dirs, get_latest_date, Filter
from eeg_tfa import Time_Frequency

###############################################################
#######                                                 #######
#######      透過滑動scale設定閥值來計算Band Power        #######
#######                                                 #######
###############################################################

def calculate_band_power(data, fs, band, threshold):
    segment_length = 2*fs
    n_per_seg=1000
    overlap = 0.2
    time = []
    fmin, fmax = band
    total_high_band_power = 0
    total_sum_psd = 0 
    segments_psd = []   # 用於存儲每個段的大於閾值的功率
    total_psd = []  # 用於存儲每個段的總功率

    fp1fp2_data = data[:,[0,1]]  #只計算fp1 & fp2 通道的資料
    print(fp1fp2_data.shape)  #(num_sample, channels) -> (123456, 2)
    
    for start in range(0, fp1fp2_data.shape[0], segment_length):
        segment = fp1fp2_data[start : start+segment_length, :]

        # 如果當前段的長度小於設定的段長度，則跳過
        if segment.shape[0]<segment_length:
            continue
        
        time.append(start / fs) # 記錄當前段的起始時間
        segment_total_psd = [] # 用於存儲當前段的總功率
        for ch in range(segment.shape[1]):
            # 計算功率譜密度（PSD）

            psd, freq = mne.time_frequency.psd_array_welch(segment[ : , ch], sfreq = fs, n_fft= 2048, n_overlap=n_per_seg//2, n_per_seg=n_per_seg)
            # Filter PSD within the frequency band
            band_psd = psd[(freq >= fmin) & (freq <= fmax)]
            band_freq = freq[(freq >= fmin) & (freq <= fmax)]
            band_power = np.sum(band_psd)  #加總這segment中的一個通道的psd
            # print('freq:',band_freq)
            # print('PSD:',band_psd)
            # print(f'{ch} band_power:{band_power}')

            #大於threshold的能量
            # print(threshold)
            print(f"Band PSD: {band_psd}, Threshold: {threshold}")
            high_power_psd = band_power[band_power > threshold]
            high_band_power = np.sum(high_power_psd)
            total_high_band_power += high_band_power
            print(f"High Band Power: {total_high_band_power}")

            # 存儲當前段的總功率和頻率
            segment_total_psd.append((band_freq, band_power))
            total_sum_psd+=band_power

        
        # 將當前段總功率添加到對應的列表中
        total_psd.append(segment_total_psd)

    print(f'total band power:{total_sum_psd}')
 
        
    return time, total_high_band_power, total_psd, total_sum_psd

def plot_psd(time, band_powers, total_psd, band, ax, threshold=None):
    fmin, fmax = band
    ax.clear()

    # fig, ax = plt.subplots(figsize=(10, 6))

    for ch in range(len(total_psd[0])):      
        powers = [segment[ch][1] for segment in total_psd]  #從每段數據中找到某個特定通道的數據。只取出這個通道的功率數據。把這些功率數據放到一個新列表裡。
        if ch == 0:
            ax.plot(time, powers, label=f'Fp1')
        if ch == 1:
            ax.plot(time, powers, label=f'Fp2')
        ax.legend(loc='upper right') 
   
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Power')
    ax.set_title(f'{band} Hz Band Power over Time')
    # ax.legend(loc='upper right')

    # Draw the threshold line if a threshold is provided
    if threshold is not None:
        ax.axhline(y=threshold, color='red', linestyle='--', label='Threshold')

    plt.tight_layout()
    plt.draw()

class EEG_GUI:
    def __init__(self, file_path):
        self.bands = {
            'Theta' : (6, 8),
            'Alpha' : (8, 12),
            'Beta' : (12, 30)
        }
        self.fs = 1000

        decoder = Decoder() 
        filter = Filter()

        eeg_data          = decoder.read_decoded(file_path)

        # truncate
        trun = 2.5e-5
        # for i in range(19): # ch1 ~ ch8
        #     eeg_data[:, i][np.where(eeg_data[:, i] >=  trun)] = trun
        #     eeg_data[:, i][np.where(eeg_data[:, i] <= -trun)] = -trun 
        
        #使用布爾演碼將不符合閥值的樣本刪除
        mask = np.any(eeg_data >= trun ,axis=1) | np.any(eeg_data <= -trun, axis = 1)
        eeg_data = eeg_data[~mask]
        
        for i in range(19): # ch1 ~ ch8
            # eeg_data[:, i] = filter.butter_bandpass_filter(eeg_data[:, i], 1, 100, 1000)
            # eeg_data[:, i] = filter.butter_bandstop_filter(eeg_data[:, i], 55, 65, 1000)
            eeg_data[:, i] = filter.butter_bandpass_filter(eeg_data[:, i], 6, 40, 1000)   
        print(eeg_data.shape)

        self.eeg_data = eeg_data[10*self.fs : -10*self.fs,:19]
        self.win_power = tk.Toplevel()
        self.win_power.title('EEG Band Power')
        self.win_power.minsize(width=1000, height=260)

        self.band_var = tk.StringVar(value='Alpha') #
        self.threshold_var = tk.DoubleVar(value=1e-14) #value=1e-11
        self.result_var_1 = tk.StringVar()
        self.result_var_2 = tk.StringVar()
        self.result_var_3 = tk.StringVar()

        #頻帶選擇(下拉式選單)
        band_label = tk.Label(self.win_power, text='選擇頻帶:')
        band_label.place(x=10, y=10)

        self.band_combobox = ttk.Combobox(self.win_power, textvariable=self.band_var)
        self.band_combobox['value'] = list(self.bands.keys())
        self.band_combobox.place(x=80, y=10)
        self.band_combobox.bind('<<ComboboxSelected>>', self.update_result)

        #閥值輸入
        threshold_label = tk.Label(self.win_power, text='輸入閥值:')
        threshold_label.place(x=10, y=35)

        self.threshold_scale = tk.Scale(self.win_power, from_=1e-10, to=1e-13, resolution=1e-13, length=200, orient='horizontal', variable=self.threshold_var, command=self.update_result)
        self.threshold_scale.place(x=80, y=35)

        # #計算Button
        # caculate_button = tk.Button(win, text='Enter', command=lambda:self.caculate(bands))
        # caculate_button.place(x=100, y=100)

        # PSD 图
        self.fig, self.ax = plt.subplots(figsize=(24, 10))
        # plt.show(block=False)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.win_power)
        self.canvas.get_tk_widget().place(x=300, y=10, width=660, height=220)

        #結果顯示
        result_label_psd = tk.Label(self.win_power, textvariable=self.result_var_1)
        result_label_psd.place(x=10, y=80)

        result_label_total = tk.Label(self.win_power, textvariable=self.result_var_2)
        result_label_total.place(x=10, y=110)

        result_label_ratio = tk.Label(self.win_power, textvariable=self.result_var_3)
        result_label_ratio.place(x=10, y=140)

        #退出介面Button
        exit_button = tk.Button(self.win_power, text='Exit', command=self.win_power.destroy)
        exit_button.place(x=130, y=180)

        self.update_result()
        self.win_power.mainloop()



    def update_result(self, *arg):
        band_name = self.band_var.get()
        threshold_value = self.threshold_var.get()
        #  tk.Label(win, text=band_name).place(x=80, y=100)
        if not band_name or threshold_value==0:
            print(band_name)
            print(threshold_value)
            messagebox.showerror("輸入錯誤", "請輸入完整頻帶和閥值")
            return

        band = self.bands[band_name]
        time ,segments_psd, total_psd, total_sum_psd= calculate_band_power(self.eeg_data, self.fs, band, threshold_value)
        # print(total_power)
        self.result_var_1.set(f'Band Power of {band_name}: {segments_psd:.2e}') #計算所有時間段中所有通道的總功率。
        self.result_var_2.set(f'Total Band Power of {band_name}: {total_sum_psd:.2e}')
        band_ratio = (segments_psd/total_sum_psd)*100
        self.result_var_3.set(f'Band Ratio of {band_name}: {band_ratio:.2f} %')
        # print({np.sum([np.sum([segment[1] for segment in seg]) for seg in segments_psd])})
        self.update_psd_plot(time ,segments_psd, total_psd, band)

    def update_psd_plot(self, time ,segments_psd, total_psd, band, *arg):
        band_name = self.band_var.get()
        threshold_value = self.threshold_var.get()
        if not band_name or not threshold_value:
            return

        band = self.bands[band_name]
        time, segments_psd, total_psd, total_sum_psd= calculate_band_power(self.eeg_data, self.fs, band, threshold_value)
        plot_psd(time, segments_psd, total_psd, band, self.ax, threshold_value)
        self.canvas.draw()

    # def band_power(self):
    #     band_name = self.band_var.get()
    #     threshold_value = self.threshold_var.get()
    #     if not band_name or not threshold_value:
    #         messagebox.showerror("輸入錯誤", "請輸入完整頻帶和閥值")
    #         return

    #     # band = self.bands[band_name]
    #     # time, segments_psd, total_psd= calculate_band_power(self.eeg_data, self.fs, band, threshold_value)
    #     self.result_var.set(f'Band Power ({band_name}): {np.sum([np.sum([segment[1] for segment in seg]) for seg in segments_psd]):.2e}')
        
    #     self.update_psd_plot()

    #     # # 绘制 PSD 图
    #     # plot_psd(segments_psd, band, self.ax, threshold_value)


if __name__ == "__main__":
    path = 'D:/GUI/rest_5/2024_05_20_1435/1.txt'
    EEG_GUI(path)

            