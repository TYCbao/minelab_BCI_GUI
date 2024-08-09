import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import pandas as pd
import mne
import os

# costum libraries
from eeg_decoder import Decoder, findall_datadate_dirs, get_latest_date, Filter

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

def plot_topology_map(file_path):
    eeg_data          = decoder.read_decoded(file_path)
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
        eeg_data[:, i] = filter.butter_bandpass_filter(eeg_data[:, i], 4, 40, 1000)     
        # eeg_data[:, i] = filter.butter_bandstop_filter(eeg_data[:, i], 55, 65, 1000)
        # eeg_data[:, i] = filter.butter_bandpass_filter(eeg_data[:, i], 8, 14, 1000)
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
    
    # 設置頭皮圖繪製參數
    # montage = plot_19ch_montage()
    montage=mne.channels.make_standard_montage('biosemi32')
    raw.set_montage(montage)
    
    # Artifact removal using ICA
    ica = mne.preprocessing.ICA(n_components=5, random_state=97, max_iter=800)
    ica.fit(raw)
    ica.exclude = [0, 1]  # Indices of the components to exclude based on visual inspection
    raw_corrected = raw.copy()
    ica.apply(raw_corrected)

    # Normalize the data
    scaler = StandardScaler()
    data_normalized = scaler.fit_transform(raw_corrected.get_data().T).T  # Normalize each channel
    raw_corrected._data = data_normalized

    # Filter the data for different frequency bands
    theta_band = raw_corrected.copy().filter(l_freq=4, h_freq=8)
    alpha_band = raw_corrected.copy().filter(l_freq=8, h_freq=12)
    beta_band = raw_corrected.copy().filter(l_freq=12, h_freq=30)

    # Compute the Power Spectral Density (PSD)
    from mne.time_frequency import psd_array_welch

    theta_psd, freqs = psd_array_welch(theta_band.get_data(), sfreq=sfreq, fmin=4, fmax=8, n_fft=2048)
    alpha_psd, freqs = psd_array_welch(alpha_band.get_data(), sfreq=sfreq, fmin=8, fmax=12, n_fft=2048)
    beta_psd, freqs = psd_array_welch(beta_band.get_data(), sfreq=sfreq, fmin=12, fmax=30, n_fft=2048)

    # Average the PSD across the frequency band
    theta_avg_psd = np.mean(theta_psd, axis=1)
    alpha_avg_psd = np.mean(alpha_psd, axis=1)
    beta_avg_psd = np.mean(beta_psd, axis=1)
    
    # Plot the topomaps
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    plt.suptitle('EEG Band Power Topography (Resting State)', fontsize=16    )

    im,_ =  mne.viz.plot_topomap(theta_avg_psd, pos=raw.info, axes=axes[0], show=False,cmap='jet')
    mne.viz.plot_topomap(alpha_avg_psd, pos=raw.info, axes=axes[1], show=False, cmap='jet')
    mne.viz.plot_topomap(beta_avg_psd, pos=raw.info, axes=axes[2], show=False, cmap='jet')


    # axes[0,0].text(0,0, f'Theta Band (4-8 Hz)',va = 'center', ha='center')
    # axes[0,1].text(0,0, f'Alpha Band (8-12 Hz)',va = 'center', ha='center')
    # axes[0,2].text(0,0, f'Beta Band (12-30 Hz)',va = 'center', ha='center')
    # 添加一个共享的 colorbar
    fig.subplots_adjust(left=0.5)
    cbar_ax = fig.add_axes([0.5, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax)

    plt.tight_layout()
    # plt.show()

    return 



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
    
    SUBJECT_NAMES =['rest_1']

    # =========================================================================== #

    # print(data_dates)  
    decoder = Decoder() 
    filter = Filter()

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
                eeg_data[:, i] = filter.butter_bandpass_filter(eeg_data[:, i], 4, 40, 1000)     
                # eeg_data[:, i] = filter.butter_bandstop_filter(eeg_data[:, i], 55, 65, 1000)
                # eeg_data[:, i] = filter.butter_bandpass_filter(eeg_data[:, i], 8, 14, 1000)
            eeg_data = eeg_data[:,:19]

            
            
            
            # #創建事件矩陣
            # # events = np.array([[int(time), 0, 1] for time in triggers])
            # # # events = np.column_stack((trigger, np.zeros(len(trigger), int), np.ones(len(triggers), int)))
            # # print(events)
   
            # # 分割數據為Epochs
            # # epochs = mne.Epochs(raw, events, event_id=1, tmin=-0.2, tmax=0.8, baseline=None, preload=True)

            # # 頻段過濾 
            # bands = {
            #     'Theta' : (4, 8),
            #     'Alpha' : (8, 12),
            #     'Beta' : (12, 30)
            # }
            # channels_of_interest = ['C3', 'C4', 'F3', 'F4']
            # # 繪製拓樸圖
            # fig, axes = plt.subplots(3, 2, figsize=(12, 20))
            # plt.suptitle('EEG Band Power Topography (Resting State)', fontsize=16    )
            # ims=[]

            # #計算每個頻段的PSD並繪製拓樸
            # for i, (band, (fmin, fmax)) in enumerate(bands.items()):
            #     #過濾頻段
            #     raw_filiter=raw.copy().filter(fmin, fmax, fir_design='firwin')

            #     psd, freq = mne.time_frequency.psd_array_welch(raw_filiter.get_data(), sfreq=sfreq, fmin=fmin, fmax=fmax, n_fft=2048)
           
            #     band_power = np.mean(psd, axis=1)
            #     print(f'{band} band power:', band_power)

            #     im,_ = mne.viz.plot_topomap(band_power, raw.info, axes=axes[i,0], names= ch_names, show=False, contours=0,cmap='jet')
            #     ims.append(im)
            #     # plt.colorbar(im, ax=axes[i,0], pad=0.01, fraction = 0.01)

            #     #繪製表格
            #     table_data = []
            #     for ch in channels_of_interest:
            #         ch_idx = ch_names.index(ch)
            #         table_data.append([ch, band_power[ch_idx]])

            #     ax_table= axes[i,1]
            #     table =ax_table.table(cellText = table_data, colLabels = ['Channel','Value'], loc='center')
            #     table.auto_set_font_size(False)
            #     table.set_fontsize(10)
            #     table.scale(1, 2)
            #     axes[i, 1].axis('off')

            # # 添加共享 colorbar
            # fig.subplots_adjust(left=0.5)
            # cbar_ax = fig.add_axes([0.9, 0.15, 0.02, 0.7])
            # fig.colorbar(ims[0], cax=cbar_ax)

            # axes[0,0].text(-0.12,0, f'Theta Band (4-8 Hz)',va = 'center', ha='center', rotation ='vertical')
            # axes[1,0].text(-0.12,0, f'Alpha Band (8-12 Hz)',va = 'center', ha='center', rotation ='vertical')
            # axes[2,0].text(-0.12,0, f'Beta Band (12-30 Hz)',va = 'center', ha='center', rotation ='vertical')
            # plt.tight_layout(rect=[0, 0, 0.85, 1])
            # plt.savefig(f"{SAVE_DIR}/{' '.join('band power topography').title()}.png", dpi = 500)
            # plt.show()
            


