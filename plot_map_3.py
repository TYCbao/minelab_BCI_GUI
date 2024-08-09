import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mne
import os

# costum libraries
from eeg_decoder import Decoder, findall_datadate_dirs, get_latest_date, Filter
from eeg_tfa import Time_Frequency

a=[]
b=[]
c=[]
cut_data = []

### 19通道 ###
def cut_data_randomhand(eeg_data, trigger, hand, tmin=-2, tmax=2):
    """    
    segment the data range from tmin to tmax based on trigger, e.g. -2 ~ 2 s
    
    Parameters
    ----------------------------------------
    `eeg_data` : eeg data with shape = (n, 10) which,
                 8 is the number of channel,
                 n is the number of sample points
    
    `tmin` : time before action, must be equal or less than 0 (<= 0)
    `tmax`: time after action, must be greater than 0 (> 0)
    

    Return
    --------------------------------------------------
    `cut_data` : segmented data in numpy.array format with shape = (trials, channels, timepoints) which,
                 trials is the number of trials,
                 channels is the number of channels (8 channels)
                 timepoints is the timepoints (length) of data specify by 'tmin' and 'tmax'
    
    `triggers` : trigger value corresponding to each trial, shape = (n, ) which n = number of triggers


    Examples
    --------------------------------------------------
    >>> cut_data, triggers = cut_data(eeg_data, tmin=-2, tmax=3, not_triggered=0)
    """    
    decoder = Decoder()
    
    fs = 1000
    tmin = tmin * fs    #1秒*1000
    tmax = tmax * fs
    # cut_data = []
    
    # print(trigger)
    # trigger_idx = decoder.find_trigger(eeg_data)
    # for idx, trigger in trigger_idx:
    #     if trigger not in trigger_to_class.keys():
    #         continue     
    
    for i ,row_data in enumerate(eeg_data): #取0.001秒的總資料
        idx=i+1 
        
        if idx == trigger and hand=='L':
            # a.append(idx)
            
            # print(eeg_data[i][:19])  #row_data=eeg_data[i]
            data_1 = eeg_data[i+tmin : i+tmax] # each data segment must be the same size !!
            a.append(data_1)
            

        elif idx == trigger and hand=='R':
            # a.append(idx)
            
            # print(eeg_data[idx][:19])  #row_data=eeg_data[i]
            data_2 = eeg_data[i+tmin : i+tmax] # each data segment must be the same size !!
            # print(data[2000])
            b.append(data_2)
            # print(cut_data)
  
        else:
            continue
        
    cut_data_l = np.array(a)
    cut_data_r = np.array(b)
    
    # print(cut_data_l.shape)
    # print(cut_data_r.shape)
    
    # cut_data_l = np.transpose(cut_data_l, (2, 0, 1))[:19] # transpose from shape = (trials, samples, channels) to (channels, trials, samples). retain only eeg channel
    # cut_data_l = np.transpose(cut_data_l, (1, 0, 2)) # transpose from shape = (channels, trials, samples) to (trials, channels, samples)
    
    # cut_data_r = np.transpose(cut_data_r, (2, 0, 1))[:19] # transpose from shape = (trials, samples, channels) to (channels, trials, samples). retain only eeg channel
    # cut_data_r = np.transpose(cut_data_r, (1, 0, 2))# transpose from shape = (channels, trials, samples) to (trials, channels, samples)
    # print(np.shape(cut_data))  #(1, 19, 6000)

    return cut_data_l,cut_data_r

### 19通道 ###
def cut_data(eeg_data, trigger,  tmin=-2, tmax=2):
    """    
    segment the data range from tmin to tmax based on trigger, e.g. -2 ~ 2 s
    
    Parameters
    ----------------------------------------
    `eeg_data` : eeg data with shape = (n, 10) which,
                 8 is the number of channel,
                 n is the number of sample points
    
    `tmin` : time before action, must be equal or less than 0 (<= 0)
    `tmax`: time after action, must be greater than 0 (> 0)
    

    Return
    --------------------------------------------------
    `cut_data` : segmented data in numpy.array format with shape = (trials, channels, timepoints) which,
                 trials is the number of trials,
                 channels is the number of channels (8 channels)
                 timepoints is the timepoints (length) of data specify by 'tmin' and 'tmax'
    
    `triggers` : trigger value corresponding to each trial, shape = (n, ) which n = number of triggers


    Examples
    --------------------------------------------------
    >>> cut_data, triggers = cut_data(eeg_data, tmin=-2, tmax=3, not_triggered=0)
    """    
    decoder = Decoder()
    
    
    fs = 1000
    tmin = tmin * fs    #1秒*1000
    tmax = tmax * fs
    # cut_data = []
    
    # print(trigger)
    # trigger_idx = decoder.find_trigger(eeg_data)
    # for idx, trigger in trigger_idx:
    #     if trigger not in trigger_to_class.keys():
    #         continue     
    
    for i ,row_data in enumerate(eeg_data): #取0.001秒的總資料
        idx=i+1 
        
        if idx == trigger:
            # a.append(idx)
            
            # print(eeg_data[i][:19])  #row_data=eeg_data[i]
            data_1 = eeg_data[i+tmin : i+tmax] # each data segment must be the same size !!
            c.append(data_1)
        else:
            continue
        
    cut_data = np.array(c)

    
    # print(cut_data_l.shape)
    # print(cut_data_r.shape)
    
    # cut_data_l = np.transpose(cut_data_l, (2, 0, 1))[:19] # transpose from shape = (trials, samples, channels) to (channels, trials, samples). retain only eeg channel
    # cut_data_l = np.transpose(cut_data_l, (1, 0, 2)) # transpose from shape = (channels, trials, samples) to (trials, channels, samples)
    
    # cut_data_r = np.transpose(cut_data_r, (2, 0, 1))[:19] # transpose from shape = (trials, samples, channels) to (channels, trials, samples). retain only eeg channel
    # cut_data_r = np.transpose(cut_data_r, (1, 0, 2))# transpose from shape = (channels, trials, samples) to (trials, channels, samples)
    # print(np.shape(cut_data))  #(1, 19, 6000)

    return cut_data

def plot_19ch_montage():
    ### 繪製通道圖 ###
    # print(mne.channels.get_builtin_montages()) 
    brain_1020_montage = mne.channels.make_standard_montage('biosemi32')
    # brain_1020_montage.plot()
    # plt.show()

    #存取通道位置
    sensor_data_32 = brain_1020_montage.get_positions()['ch_pos']
    sensor_dataframe_32 = pd.DataFrame(sensor_data_32).T
    # print(sensor_dataframe_32)
    # sensor_dataframe_32.to_excel('sensor_dataframe_32.xlsx')

    ###匯入自行定義之通道座標
    sensor_data_19 = pd.read_excel('sensor_dataframe_19.xlsx', index_col=0)
    ch_names = np.array(sensor_data_19.index)
    position = np.array(sensor_data_19)
    sensorPosition = dict(zip(ch_names, position))
    montage_19ch = mne.channels.make_dig_montage(ch_pos=sensorPosition)
    # montage_19ch.plot()
    # plt.show()

    return montage_19ch


if __name__ == "__main__":
    # =========================================================================== #
    #                            [Parameter setting]                            
    # =========================================================================== #
    # [data epoching parameters]  
    # (string)      | EXP_DIR          : 受試者資料夾，其中包括多個腦波資料夾  
    # (list)        | data_dates       : 受試者資料夾中的腦波資料夾名稱
    # (dict)        | trigger_to_class : 腦波機trigger對應的類別名稱  
    # (int/fp)      | tmin, tmax       : 動作區間[tmin, tmax]以秒為單位  
    #
    # [ERDS parameters]
    # (list/string) | freq_band        : 濾波範圍      
    # (int/fp)      | rest_start       : 休息開始時間點(已經切好的epoch為基準)，以秒為單位
    # (int/fp)      | rest_end         : 休息結束時間點(已經切好的epoch為基準)，以秒為單位
    # (int/fp)      | onset            : 動作起始點，以秒為單位  
    # (string)      | data_type        : 要看的資料類別，"AO"、"MI"或"ALL"
    # (string)      | SAVE_DIR         : 儲存ERDS圖片資料夾
    # =========================================================================== #
    # name = 均 彥 哲 皓 瑞
    name = "皓"

    EXP_DIR          = r'd:\EEG_data_recv_1\23121103'
    # EXP_DIR          = "H:/Chengfu/EEG_dataset/4-classes-AO-MI/specific-subject/富"
    #### EXP_DIR          = f"H:/Chengfu/EEG_dataset/4-classes-AO-MI/offline/{name}"
    #### data_dates       = findall_datadate_dirs(EXP_DIR) # data_dates = ['2023_02_10_1434']
    # data_date = get_latest_date(EXP_DIR)
    # data_dates = ['2023_12_11_1538']
    
    trigger_to_class = {254: "left_hand", 253: "right_hand", 252: "both_hand", 251: "both_feet"}
    trigger_to_class = {254: "left_hand", 253: "right_hand", 251: "both_feet"}
    tmin, tmax       = -2, 6 # 區間為trigger前2秒到trigger後4秒, 以秒為單位
    freq_band        = [8, 14]# or freq_band  = 'alpha'
                      
    rest_start       = 0
    rest_end         = 2 # rest interval [0 - 2] s
    onset            = 2 # movement onset, movement start at 2s
    data_type        = "AO" # ALL AO MI 
    # SAVE_DIR         = f'./ERDS_figs/{name}/{data_type}'
    # SAVE_DIR         =  f'{EXP_DIR}/{data_dates[0]}'
    # SAVE_DIR         = f'./2023_10_20_1716'
    
    SUBJECT_NAMES =['rest_6']

    # =========================================================================== #

    # print(data_dates)  
    decoder = Decoder() 
    filter = Filter()
    tfa = Time_Frequency()

    for subject in SUBJECT_NAMES:
        # Read eeg data
        EXP_DIR     = f"D:/eeg_code_for_autoecnoder/EEG_dataset/2-classes-AO-MI/offline/{subject}"
        data_dates  = findall_datadate_dirs(EXP_DIR)
        SAVE_DIR    = f"{EXP_DIR}/{data_dates}" 
        if not (os.path.isdir(SAVE_DIR)): os.mkdir(SAVE_DIR)
        else: print(f"'{SAVE_DIR}' already exists! it will be overwrite")


        for i, data_date in enumerate(data_dates):
            print(f"{i+1} / {len(data_dates)}")
            decoded_file_path = f'{EXP_DIR}/{data_date}/1.txt'
            # trigger_file_path = f'{EXP_DIR}/{data_date}/tri_1.txt'
            # if "1.txt"and "tri_1.txt" not in os.listdir(f'{EXP_DIR}/{data_date}'):
            #     # 如果不存在，跳过当前实验文件夹
            #     continue
            eeg_data          = decoder.read_decoded(decoded_file_path)
            # triggers          = decoder.read_trigger_all(trigger_file_path)

            # trigger=[]
            # for i in triggers:
            #     tri = int(i)
            #     trigger.append(tri)
            # trigger = np.array(trigger)

            # truncate
            trun = 2.5e-5
            for i in range(19): # ch1 ~ ch8
                eeg_data[:, i][np.where(eeg_data[:, i] >=  trun)] = trun
                eeg_data[:, i][np.where(eeg_data[:, i] <= -trun)] = -trun 

            for i in range(19): # ch1 ~ ch8
                eeg_data[:, i] = filter.butter_bandpass_filter(eeg_data[:, i], 1, 100, 1000)
                eeg_data[:, i] = filter.butter_bandstop_filter(eeg_data[:, i], 55, 65, 1000)
                eeg_data[:, i] = filter.butter_bandpass_filter(eeg_data[:, i], 6, 40, 1000)     
                
                
            eeg_data = eeg_data[:,:19]
            
            # 設置基本參數
            sfreq = 1000  # 假設采樣率為1000 Hz
            n_channels = eeg_data.shape[1]
            n_samples = eeg_data.shape[0]
            ch_types=['eeg']*19
            print('eeg data shape:',eeg_data.shape)
            print('channels:',n_channels)
            ch_names = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T7', 'C3', 'Cz', 'C4'
                        ,'T8' , 'P7',  'P3', 'Pz', 'P4', 'P8', 'O1', 'O2'][:n_channels]
            info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
            
            
            # 創建RawArray
            raw = mne.io.RawArray(eeg_data.T, info)

            # 选择EEG通道
            picks = mne.pick_types(raw.info, meg=False, eeg=True, eog=False, stim=False)
            
            # 設置頭皮圖繪製參數
            # montage = plot_19ch_montage()
            montage=mne.channels.make_standard_montage('biosemi32')
            raw.set_montage(montage)
 
            # 頻段過濾 
            bands = {
                'Theta' : (4, 8),
                'Alpha' : (8, 12),
                'Beta' : (12, 30)
            }
            channels_of_interest = ['C3', 'C4', 'F3', 'F4']
            # 繪製拓樸圖
            fig, axes = plt.subplots(3, 3, figsize=(6, 10))
            fig.subplots_adjust(wspace=0.3)

            plt.suptitle('EEG Band Power Topography (Resting State)', fontsize=16)

            fft_data = np.zeros((n_samples // 20, n_channels))
            for channel in range(n_channels):
                y = eeg_data[:, channel]
                t,freq,Y =tfa.fft(y)
                print(freq)
                fft_data[:,channel] = np.abs(Y)

            #計算每個頻段的FFT並繪製拓樸
            for i, (band, (fmin, fmax)) in enumerate(bands.items()):
                #過濾頻段
                # raw_filiter=raw.copy().filter(fmin, fmax, fir_design='firwin')
                idx_band = np.logical_and(freq >= fmin, freq <= fmax)
                fft_band = fft_data[idx_band,: ].mean(axis=0)  # 在频带内取平均值
                print(f'頻帶{band}:',fft_band)

            #     # psd, freq = mne.time_frequency.psd_array_welch(raw_filiter.get_data(), sfreq=sfreq, fmin=fmin, fmax=fmax, n_fft=2048)
                
            #     band_power = np.mean(psd, axis=1)
            #     print(f'{band} band power:', band_power)

                im,_ = mne.viz.plot_topomap(fft_band, raw.info, axes=axes[i,0], names= ch_names, show=False, contours=0,cmap='jet',vlim=(2e-8, 1.5e-7))# 

                # plt.colorbar(im, ax=axes[i,0], pad=0.01, fraction = 0.01)

                #繪製表格
                table_data = []
                for ch in channels_of_interest:
                    ch_idx = ch_names.index(ch)
                    table_data.append([ch, fft_band[ch_idx]])

                ax_table= axes[i,2]
                table =ax_table.table(cellText = table_data, colLabels = ['Channel','Value'], loc='center', cellLoc='center')
                table.auto_set_font_size(False)
                table.set_fontsize(10)
                table.scale(1, 2)
                axes[i, 1].set_aspect(2)
                axes[i, 2].axis('off')
                axes[i, 1].axis('off')


            c_bar = fig.colorbar(im, ax=axes[1,1], fraction=0.05, location='left')
            # 调整颜色条的大小和字体大小
            c_bar.ax.tick_params(labelsize=8)  # 设置颜色条刻度标签字体大小
            c_bar.ax.yaxis.label.set_fontsize(8)  # 设置颜色条标签（如果有的话）的字体大小

            # 设置颜色条宽度和高度
            c_bar.ax.figure.set_size_inches(15, 20)  # 设置颜色条的宽度和高度

            # # 添加共享 colorbar
            # fig.subplots_adjust(right=0.8)
            # cbar_ax = fig.add_axes([0.9, 0.15, 0.015, 0.7])
            # fig.colorbar(im, cax=cbar_ax, ax=axes, shrink=0.5)

            axes[0,0].text(-0.12,0, f'Theta Band (4-8 Hz)',va = 'center', ha='center', rotation ='vertical')
            axes[1,0].text(-0.12,0, f'Alpha Band (8-12 Hz)',va = 'center', ha='center', rotation ='vertical')
            axes[2,0].text(-0.12,0, f'Beta Band (12-30 Hz)',va = 'center', ha='center', rotation ='vertical')
            
            plt.tight_layout(rect=[0, 0, 1, 1])
            plt.savefig(f"{SAVE_DIR}/{' '.join('band power topography_FFT').title()}.png", dpi = 500)
            plt.show()
            


