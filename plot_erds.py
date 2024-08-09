#%%
# import libraries
import copy
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.integrate
from scipy.stats import pearsonr
import os

# costum libraries
from eeg_decoder import Decoder, findall_datadate_dirs, get_latest_date, Filter
from eeg_erds import ERDS

a=[]
b=[]
cut_data = []
'''
### 8通道 ###
def cut_data(eeg_data, trigger_to_class, tmin=-2, tmax=2): 
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
    cut_data = []
    triggers = []

    trigger_idx = decoder.find_trigger(eeg_data)
    for idx, trigger in trigger_idx:
        if trigger not in trigger_to_class.keys():
            continue        
        data = eeg_data[idx+tmin : idx+tmax] # each data segment must be the same size !!
        cut_data.append(data)
        triggers.append(trigger) 

    
    # trun into np array
    triggers = np.array(triggers)
    cut_data = np.array(cut_data)
    # cut_data = np.transpose(cut_data, (2, 0, 1))[:8] # transpose from shape = (trials, samples, channels) to (channels, trials, samples). retain only eeg channel
    # cut_data = np.transpose(cut_data, (1, 0, 2)) # transpose from shape = (channels, trials, samples) to (trials, channels, samples)

    return cut_data, triggers
'''
### 19通道 ###
def cut_data(eeg_data, trigger, hand, tmin=-2, tmax=2):
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
    # a=[]
    # b=[]
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

    
    # trun into np array
    
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
    
def erd_value( data, band, freq_band):  
    my_erds = ERDS(rest_start=rest_start, rest_end=rest_end, onset=onset)
    erds_hand    = my_erds.get_erds(data, freq_band=freq_band)
    print(erds_hand.shape)
    # my_erds.plot_erds_19(erds_hand, title= f"{hand} hand ERD/ERS".title() , save_dir = SAVE_DIR)
    
    #存取excel檔
    # df_R=pd.DataFrame(erds_hand)
    # excel_path = f'{SAVE_DIR}/erds_right_output.xlsx'  # 定义 Excel 文件路径
    # df_R.to_excel(excel_path, index=False)  # 如果不想保留索引，设置 index=False


    ### 想像運動 ###
    channel_dict={'F3':3, 'C3':8, 'P3':13, 'F4':5, 'C4':10, 'P4':15}

    tabel_data=[]
    mean=[]
    fig, ax = plt.subplots(figsize=(5, 9))  # 调整图表大小以适应表格
    ax.axis('off')  # 隐藏坐标轴    
    for key, index in channel_dict.items():
        erd_4s=erds_hand[index,2000:6000]  #2000:6000代表trigger到後4秒的資料
        time_4s = np.linspace(0, 4, 4000)  # 0到4秒，共4000点

        #提取小於0的值(ERD)
        negative_indices = erd_4s < 0
        negative_val = np.where(negative_indices, erd_4s, 0)  

        #提取大於0的值(ERS)
        positive_indices = erd_4s > 0
        positive_val = np.where(positive_indices, erd_4s, 0)  

        
        negative_value = erd_4s[erd_4s < 0]
        # print(len(right_negative_value))
        average_of_negatives_value=np.mean(negative_value)
        # print(f'通道:{key}')
        # print(f'mean:', average_of_negatives_value)
        mean.append(round(average_of_negatives_value,2))
        std_of_negatives_value=np.std(negative_value)  
        # print(f'std:', std_of_negatives_value)
        max_negatives_value=np.min(negative_value)
        # print(f'max amp:', max_negatives_value)

        erd_area = scipy.integrate.simpson(negative_val, time_4s)  # 使用 Simpson 積分計算面積
        # print("ERD積分後的面積:",np.abs(erd_area))

        ers_area = scipy.integrate.simpson(positive_val, time_4s)  # 使用 Simpson 積分計算面積
        # print("ERS積分後的面積:",np.abs(ers_area),'\n')
        # print(erds_right[5])
        # print(erds_right.shape)    
        data = [f'{key}',"mean:"+f"{average_of_negatives_value:.2f}\n"
                            "std:"+f"{std_of_negatives_value:.2f}\n"
                            "max amp:"+f"{max_negatives_value:.2f}\n"
                            "ERD area:"+ f"{np.abs(erd_area):.2f}\n"
                            "ERS area:"+ f"{np.abs(ers_area):.2f}"]
        tabel_data.append(data)
                
    # 使用 plt.table() 创建表格
    table = plt.table(cellText=tabel_data, colLabels=None, cellLoc='center', loc='center')

    # 调整表格样式
    plt.suptitle(f'{band} band ERD/ERS Value')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(0.8, 8)  # 调整表格的比例
    # print(f'{band}的平均值:', mean, '資料長度:', len(mean) )



    # 显示表格
    plt.show()
    return average_of_negatives_value, std_of_negatives_value, max_negatives_value

def erd_value_with_file(file_path):  
    decoder = Decoder() 
    filter = Filter()
    decoded_file_path = f'{file_path}/1.txt' # 1.txt
    trigger_file_path = f'{file_path}/tri_1.txt'
    print(decoded_file_path)
    # decoded_file_path = f'{EXP_DIR}/{data_dates[i]}/1/1.npy' # 1.txt
    eeg_data          = decoder.read_decoded(decoded_file_path)


    trigger     = decoder.read_trigger_all(trigger_file_path)
    # print(trigger_left, trigger_right)

    # truncate
    trun = 2.5e-5
    for i in range(19): # ch1 ~ ch8
        eeg_data[:, i][np.where(eeg_data[:, i] >=  trun)] = trun
        eeg_data[:, i][np.where(eeg_data[:, i] <= -trun)] = -trun 

    for i in range(19): # ch1 ~ ch8
        # eeg_data[:, i] = filter.butter_bandpass_filter(eeg_data[:, i], 1, 100, 1000)     
        # eeg_data[:, i] = filter.butter_bandstop_filter(eeg_data[:, i], 55, 65, 1000)
        eeg_data[:, i] = filter.butter_bandpass_filter(eeg_data[:, i], 4, 40, 1000)  

    print(f"濾波後資料形狀：{eeg_data.shape}")
    triggers = []         
    for j, tri in enumerate(trigger): 
        if (int(eeg_data.shape[0])-int(tri)) >= 6000:
            triggers.append(tri) 
            # print(triggers)
            # print(j,trigger_id)
            ### cut data ###
            data,r    = cut_data(eeg_data, 
                                tmin = -2, 
                                tmax = 6, 
                                trigger = int(tri),
                                hand = 'L')

    data = np.transpose(data, (2, 0, 1))[:19] # transpose from shape = (trials, samples, channels) to (channels, trials, samples). retain only eeg channel
    data = np.transpose(data, (1, 0, 2)) # transpose from shape = (channels, trials, samples) to (trials, channels, samples)
    print(data.shape)
    
    triggers = np.array(triggers)
    print(f"\nshape of cut data = {data.shape}, trigger = {triggers.shape}, trigger value = {triggers}")
    
    print('-'*70)  



# =========================================================================== #
#                        [Calculate & plot ERD/ERS]                         
# =========================================================================== #    
    print("Plot ERD/ERS")
    
    bands = {
                'Alpha' : [8, 12],
                'Beta' : [12, 30]
            }
    # 初始化圖像變數
    fig_alpha, fig_beta = None, None

    my_erds = ERDS(rest_start=0, rest_end=2, onset=2)
    for band, value in bands.items():
        print(band, value)
        erds_hand    = my_erds.get_erds(data, freq_band=value)
        
        print(erds_hand.shape)
        # my_erds.plot_erds_19(erds_hand, title= f"right hand".title() , save_dir = file_path)
        

        ### 想像運動 ###
        channel_dict={'F3':3, 'C3':8, 'P3':13, 'F4':5, 'C4':10, 'P4':15}

        tabel_data=[]
        mean=[]
           
        for key, index in channel_dict.items():
            erd_4s=erds_hand[index,2000:6000]  #2000:6000代表trigger到後4秒的資料
            time_4s = np.linspace(0, 4, 4000)  # 0到4秒，共4000点

            #提取小於0的值(ERD)
            negative_indices = erd_4s < 0
            negative_val = np.where(negative_indices, erd_4s, 0)  

            #提取大於0的值(ERS)
            positive_indices = erd_4s > 0
            positive_val = np.where(positive_indices, erd_4s, 0)  

            
            negative_value = erd_4s[erd_4s < 0]
            # print(len(right_negative_value))
            average_of_negatives_value=np.mean(negative_value)
            # print(f'通道:{key}')
            # print(f'mean:', average_of_negatives_value)
            mean.append(round(average_of_negatives_value,2))
            std_of_negatives_value=np.std(negative_value)  
            # print(f'std:', std_of_negatives_value)
            max_negatives_value=np.min(negative_value)
            # print(f'max amp:', max_negatives_value)

            erd_area = scipy.integrate.simpson(negative_val, time_4s)  # 使用 Simpson 積分計算面積
            # print("ERD積分後的面積:",np.abs(erd_area))

            ers_area = scipy.integrate.simpson(positive_val, time_4s)  # 使用 Simpson 積分計算面積
            # print("ERS積分後的面積:",np.abs(ers_area),'\n')
            # print(erds_right[5])
            # print(erds_right.shape)    
            band_data = [f'{key}',"mean:"+f"{average_of_negatives_value:.2f}\n"
                                "std:"+f"{std_of_negatives_value:.2f}\n"
                                "max amp:"+f"{max_negatives_value:.2f}\n"
                                "ERD area:"+ f"{np.abs(erd_area):.2f}\n"
                                "ERS area:"+ f"{np.abs(ers_area):.2f}"]
            tabel_data.append(band_data)
        # print(f'{band}的平均值:', mean, '資料長度:', len(mean) )

        if band == 'Alpha':
            fig_alpha, ax = plt.subplots(figsize=(3, 9))  # 调整图表大小以适应表格
            ax.axis('off')  # 隐藏坐标轴 

            # 使用 plt.table() 创建表格
            table = plt.table(cellText=tabel_data, colLabels=None, cellLoc='center', loc='center')
            
            # 调整表格样式
            plt.suptitle(f'{band} band ERD/ERS Value')
            table.auto_set_font_size(False)
            table.set_fontsize(12)
            table.scale(0.95, 8)  # 调整表格的比例
        if band == 'Beta':
            fig_beta, ax = plt.subplots( figsize=(3, 9))  # 调整图表大小以适应表格
            ax.axis('off')  # 隐藏坐标轴 
                    
            # 使用 plt.table() 创建表格
            table = plt.table(cellText=tabel_data, colLabels=None, cellLoc='center', loc='center')
            
            # 调整表格样式
            plt.suptitle(f'{band} band ERD/ERS Value')
            table.auto_set_font_size(False)
            table.set_fontsize(12)
            table.scale(0.95, 8)  # 调整表格的比例

    return fig_alpha, fig_beta

def create_erd_image(file_path, band):
    decoder = Decoder() 
    filter = Filter()
    decoded_file_path = f'{file_path}/1.txt' # 1.txt
    trigger_file_path = f'{file_path}/tri_1.txt'
    print(decoded_file_path)
    # decoded_file_path = f'{EXP_DIR}/{data_dates[i]}/1/1.npy' # 1.txt
    eeg_data          = decoder.read_decoded(decoded_file_path)
    print(eeg_data.shape)
    trigger     = decoder.read_trigger_all(trigger_file_path)
    
    # truncate
    trun = 2.5e-5
    for i in range(19): # ch1 ~ ch8
        eeg_data[:, i][np.where(eeg_data[:, i] >=  trun)] = trun
        eeg_data[:, i][np.where(eeg_data[:, i] <= -trun)] = -trun 

    for i in range(19): # ch1 ~ ch8
        # eeg_data[:, i] = filter.butter_bandpass_filter(eeg_data[:, i], 1, 100, 1000)     
        # eeg_data[:, i] = filter.butter_bandstop_filter(eeg_data[:, i], 55, 65, 1000)
        eeg_data[:, i] = filter.butter_bandpass_filter(eeg_data[:, i], 4, 40, 1000)  
    triggers = []         
    for j, tri in enumerate(trigger): 
        if (int(eeg_data.shape[0])-int(tri)) >= 6000:
            triggers.append(tri) 

            #print(triggers_L)
            # print(j,trigger_id)
            ### cut data ###
            data,r    = cut_data(eeg_data, 
                                tmin = -2, 
                                tmax = 6, 
                                trigger = int(tri),
                                hand = 'L')
            # print(data.shape)

    data = np.transpose(data, (2, 0, 1))[:19] # transpose from shape = (trials, samples, channels) to (channels, trials, samples). retain only eeg channel
    data = np.transpose(data, (1, 0, 2)) # transpose from shape = (channels, trials, samples) to (trials, channels, samples)
    print(data.shape)
    
    triggers = np.array(triggers)
    print(f"\nshape of cut data = {data.shape}, trigger = {triggers.shape}, trigger value = {triggers}")
    
    print('-'*70)  
    bands = {
                'Alpha' : [8, 12],
                'Beta' : [12, 30]
            }
    

    my_erds = ERDS(rest_start=0, rest_end=2, onset=2)
    erds    = my_erds.get_erds(data, freq_band=bands[band])
    fig_erd = my_erds.plot_erds_19(erds, title= f'{band} band'.title() + ' ERD/ERS', save_dir = file_path)

    # if band == 'beta':
    #     my_erds = ERDS(rest_start=rest_start, rest_end=rest_end, onset=onset)
    #     erds    = my_erds.get_erds(data, freq_band=bands['Beta'])
    #     fig_erd = my_erds.plot_erds_19(erds, title= 'Beta band'.title() + ' ERD/ERS', save_dir = file_path)

    return fig_erd


    
    



# %%
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
    
    EXP_DIR          = r'd:\user\Desktop\minelab\EEG_data_recv\240105007'
    # EXP_DIR          = "H:/Chengfu/EEG_dataset/4-classes-AO-MI/specific-subject/富"
    #### EXP_DIR          = f"H:/Chengfu/EEG_dataset/4-classes-AO-MI/offline/{name}"
    #### data_dates       = findall_datadate_dirs(EXP_DIR) # data_dates = ['2023_02_10_1434']
    # data_date = get_latest_date(EXP_DIR)
    # data_dates = ['2024_01_12_1111']
    
    trigger_to_class = {254: "left_hand", 253: "right_hand", 252: "both_hand", 251: "both_feet"}
    trigger_to_class = {254: "left_hand", 253: "right_hand", 251: "both_feet"}
    tmin, tmax       = -2, 6 # 區間為trigger前2秒到trigger後4秒, 以秒為單位
    bands = {
                'Alpha' : [8, 12],
                'Beta' : [12, 30]
            }
    freq_band        = [8,14] # or freq_band  = 'alpha'
    rest_start       = 0
    rest_end         = 2 # rest interval [0 - 2] s
    onset            = 2 # movement onset, movement start at 2s
    data_type        = "AO" # ALL AO MI 
    # SAVE_DIR         = f'./ERDS_figs/{name}/{data_type}'
    # SAVE_DIR         =  f'{EXP_DIR}/{data_dates[0]}'
    # SAVE_DIR         =  r'd:\user\Desktop\minelab\EEG_data_recv\23111501\2023_11_15_1434'

    SUBJECT_NAMES =['240417030']

    # =========================================================================== #

    # print(data_dates)  
    decoder = Decoder() 
    filter = Filter()
    file_path = r'D:\eeg_code_for_stroke_patients\EEG_dataset\2-classes-AO-MI\offline'

    # fig_alpha, fig_beta = erd_value_with_file(file_path)
    # # 確保圖像顯示
    # if fig_alpha is not None:
    #     fig_alpha.show()
    # if fig_beta is not None:
    #     fig_beta.show()

    # plt.show()  # 顯示所有圖像
    for subject in SUBJECT_NAMES:
        # Read eeg data
        EXP_DIR     = f"D:/eeg_code_for_stroke_patients/EEG_dataset/2-classes-AO-MI/offline/{subject}"
        data_dates  = findall_datadate_dirs(EXP_DIR)    

         
        # Read eeg data and cut data based on trigger 
        for i ,data_date in enumerate(data_dates): 
            print(f"{i+1} / {len(data_dates)}")
            ### 讀取解碼後的EEG資料，1.txt(1.npy) ### 
            SAVE_DIR    = f"{EXP_DIR}/{data_dates[i]}" 
            if not (os.path.isdir(SAVE_DIR)): os.mkdir(SAVE_DIR)
            else: print(f"'{SAVE_DIR}' already exists! it will be overwrite") 

            if "1.txt"and "tri_1.txt" not in os.listdir(f'{EXP_DIR}/{data_date}'):
                # 如果不存在，跳过当前实验文件夹
                continue
            
            decoded_file_path = f'{EXP_DIR}/{data_dates[i]}/1.txt' # 1.txt
            trigger_file_path = f'{EXP_DIR}/{data_dates[i]}/tri_1.txt'
            
            print(decoded_file_path)
            # decoded_file_path = f'{EXP_DIR}/{data_dates[i]}/1/1.npy' # 1.txt
            eeg_data          = decoder.read_decoded(decoded_file_path)
            print(eeg_data.shape)
            trigger     = decoder.read_trigger_all(trigger_file_path)
            # print(trigger_left, trigger_right)
            

            # truncate
            trun = 2.5e-5
            for i in range(19): # ch1 ~ ch8
                eeg_data[:, i][np.where(eeg_data[:, i] >=  trun)] = trun
                eeg_data[:, i][np.where(eeg_data[:, i] <= -trun)] = -trun 

            for i in range(19): # ch1 ~ ch8
                # eeg_data[:, i] = filter.butter_bandpass_filter(eeg_data[:, i], 1, 100, 1000)     
                # eeg_data[:, i] = filter.butter_bandstop_filter(eeg_data[:, i], 55, 65, 1000)
                eeg_data[:, i] = filter.butter_bandpass_filter(eeg_data[:, i], 4, 40, 1000)  
            ### ----------------------------------------------------------------------- ###

            ### cut data ###
            # data, triggers    = cut_data(eeg_data, 
            #                             tmin = tmin, 
            #                             tmax = tmax, 
            #                             trigger_to_class = trigger_to_class)    
            
            ### stack each of the different class of data to data_dict ###

            # trigger = trigger_data
            '''
            count = 0
            for i, trigger in enumerate(triggers):
                if trigger in trigger_to_class.keys():                
                    class_name = trigger_to_class[trigger]                                                        
                    if data_type == "ALL":
                        data_dict[class_name] = np.append(data_dict[class_name], np.expand_dims(data[i], axis=0), axis = 0)

                    elif data_type == "AO":
                        # 只抓AO或MI資料
                        if count < 2:
                            data_dict[class_name] = np.append(data_dict[class_name], np.expand_dims(data[i], axis=0), axis = 0)
                            count += 1
                        else:
                            count = 0

                    elif data_type == "MI":
                        if count < 2:
                            count += 1
                        else:
                            data_dict[class_name] = np.append(data_dict[class_name], np.expand_dims(data[i], axis=0), axis = 0)
                            count = 0
            '''
            triggers = []         
            for j, tri in enumerate(trigger): 
                triggers.append(tri)
                #print(triggers_L)
                # print(j,trigger_id)
                ### cut data ###
                data, _ = cut_data(eeg_data, 
                                    tmin = tmin, 
                                    tmax = tmax, 
                                    trigger = int(tri),
                                    hand = 'L')
                # print(data.shape)

            data = np.transpose(data, (2, 0, 1))[:19] # transpose from shape = (trials, samples, channels) to (channels, trials, samples). retain only eeg channel
            data = np.transpose(data, (1, 0, 2)) # transpose from shape = (channels, trials, samples) to (trials, channels, samples)
            print(data.shape)
            
            triggers = np.array(triggers)
            # print(triggers)
            
            # 根據trigger_to_class中的value(類別名稱)來創建data_dict字典 
            # 建好的字典 : data_dict = {'right_hand' : data, 'left_hand' : data} 
            num_samples = (tmax - tmin) * 1000 # fs = 1000
            data_dict = {}
            for value in trigger_to_class.values(): # value = "right_hand", "left_hand",...
                data_dict[value] = np.zeros((0, 19, num_samples)) # 創建空的三維數組
            
            # count = 0
            # for i, trigger in enumerate(triggers):
                            
            #     class_name = 'left_hand'
            #     # print(data_type)                                                      
            #     if data_type == "ALL":
            #         data_dict[class_name] = np.append(data_dict[class_name], np.expand_dims(data[i], axis=0), axis = 0)

            #     elif data_type == "AO":
            #         # 只抓AO或MI資料
            #         if count < 2:
            #             data_dict[class_name] = np.append(data_dict[class_name], np.expand_dims(data[i], axis=0), axis = 0)
            #             count += 1
            #         else:
            #             count = 0

            #     elif data_type == "MI":
            #         if count < 2:
            #             count += 1
            #         else:
            #             data_dict[class_name] = np.append(data_dict[class_name], np.expand_dims(data[i], axis=0), axis = 0)
            #             count = 0


                ### stack each of the different class of data to data_dict ###

            # for i ,row_data in enumerate(eeg_data[:1]): #取0.001秒的總資料
            #     print(i+1,row_data)
                #     if trigger==row_data:
                #         # for k,data in enumerate(row_data[:19]):  #取19通道
                #         print("#####")
                #     else:
                #         pass
                        

                
            
            print(f"\nshape of cut data = {data.shape}, trigger = {triggers.shape}, trigger value = {triggers}")

            # print(f"\nshape of cut left_hand data = {data_left.shape}, trigger = {triggers_L.shape}, trigger value = {triggers_L}")
            # print(f"\nshape of cut right_hand data = {data_right.shape}, trigger = {triggers_R.shape}, trigger value = {triggers_R}")

            
            print('-'*70)  


    # =========================================================================== #
    #                        [Calculate & plot ERD/ERS]                         
    # =========================================================================== #    
            print("Plot ERD/ERS")
            # print(data)
            for key, value in bands.items():
                erd_value( data, key, value)
            
    # for key in data_dict.keys(): # "right_hand", "left_hand", ...
        
    #     ### 取得某一類別的腦波資料 ###
    #     # shape = (trials, channels, timepoints)
    #     data = data_dict[key] 

    #     # data = data[:, 2:5, :] # 只取C3、Cz、C4通道的資料
    #     print(f"{key:<10} = {data.shape}")
        
    
    
        # calculate ERD/ERS and plot ###
        my_erds = ERDS(rest_start=rest_start, rest_end=rest_end, onset=onset)
        erds    = my_erds.get_erds(data, freq_band=freq_band)
        my_erds.plot_erds_19(erds, title= ' '.join(key.split('_')).title() + ' ERD/ERS', save_dir = SAVE_DIR)


    # for i in range(16):

    # 指定要寫入的檔案名稱
    # file_path = SAVE_DIR + "output_original.txt"
    # with open(file_path, 'a') as file:
    #         # 將資料寫入檔案
    #         file.write('\t'+'mean'+'\t'+'std'+'\t'+'amp'+"\n")
    
    # my_erds = ERDS(rest_start=rest_start, rest_end=rest_end, onset=onset)
    # erds_left    = my_erds.get_erds(data_left, freq_band=freq_band)
    # # my_erds.plot_erds_19(erds_left, title= 'Left Hand ERD/ERS '.title() , save_dir = SAVE_DIR)
    
    # df_L=pd.DataFrame(erds_left)
    # excel_path = f'{SAVE_DIR}/erds_left_output.xlsx'  # 定义 Excel 文件路径
    # df_L.to_excel(excel_path, index=False)  # 如果不想保留索引，设置 index=False

    # ### 舉左手想像運動 ###
    # left_hand_channel=[3,8,13,5,10,15]
    # left_mean=[]
    
    # for i in left_hand_channel:
    #     erd_left_2s=erds_left[i,2000:6000] #取從trigger後4秒內的資料
    #     left_negative_value = erd_left_2s[erd_left_2s<0] #取ERD，等於小於0的值
    #     # print(left_negative_value)
    #     left_average_of_negatives_value=np.mean(left_negative_value)
    #     print('left mean:', left_average_of_negatives_value)
    #     left_mean.append(round(left_average_of_negatives_value,2))
    #     left_std_of_negatives_value=np.std(left_negative_value)
    #     print('left std:' , left_std_of_negatives_value)
    #     left_max_negatives_value=np.min(left_negative_value)
    #     print('left max amp:', left_max_negatives_value)

    #     # with open(file_path, 'a') as file:
    #     #     # 將資料寫入檔案
    #     #     # file.write(str(i) + "\t"+'left mean' +':' +str(round(left_average_of_negatives_value,2) + "\t" +'left std' +':'+str(left_std_of_negatives_value)+ "\t" +'left max amp:'+str(left_max_negatives_value) +"\n")
    #     #     file.write('left' +str(i) +':' +'\t'+str(round(left_average_of_negatives_value,2)) + "\t" +str(round(left_std_of_negatives_value,2))+ "\t"+str(round(left_max_negatives_value,2)) +"\n")
    # print(left_mean)
    # print(len(left_mean))

    # erds_right    = my_erds.get_erds(data_right, freq_band=freq_band)
    # # my_erds.plot_erds_19(erds_right, title= 'Right Hand ERD/ERS '.title() , save_dir = SAVE_DIR)
    # df_R=pd.DataFrame(erds_right)
    # excel_path = f'{SAVE_DIR}/erds_right_output.xlsx'  # 定义 Excel 文件路径
    # df_R.to_excel(excel_path, index=False)  # 如果不想保留索引，设置 index=False


    # ### 舉右手想像運動 ###
    # right_hand_channel=[3,8,13,5,10,15]
    # right_mean=[]
    # for i in right_hand_channel:
    #     erd_right_2s=erds_right[i,2000:6000]
    #     right_negative_value = erd_right_2s[erd_right_2s<0]
    #     # print(len(right_negative_value))
    #     right_average_of_negatives_value=np.mean(right_negative_value)
    #     print('right mean:', right_average_of_negatives_value)
    #     right_mean.append(round(right_average_of_negatives_value,2))
    #     right_std_of_negatives_value=np.std(right_negative_value)  
    #     print('right std:', right_std_of_negatives_value)
    #     right_max_negatives_value=np.min(right_negative_value)
    #     print('right max amp:', right_max_negatives_value)
    #     # print(erds_right[5])
    #     # print(erds_right.shape)
    # print(right_mean)
    # print(len(right_mean))







    # # clinical_scores= np.array([59,105,1,2,2,2,2,2,0,11,0,1,0,12,16,24,28,15,1])
    # # df = pd.DataFrame({'Left_Mean': left_mean, 'Right_Mean': right_mean})
    # #     #,2,2,0,11,0,1,0,12,16,24,28,15,1
    # # df.to_excel('output.xlsx', index=False)
    
    # # # 確認左右手 ERD 數據的長度相同
    # # if len(left_mean) == len(right_mean):
    # #     # 使用迴圈計算每個臨床分數的相關性
    # #     for i, clinical_score in enumerate(clinical_scores):
    # #         if i < len(left_mean):  # 確保索引不越界
    # #             # 計算左手 ERD 與臨床分數的相關性
    # #             correlation_left, p_value = pearsonr([left_mean[i]], [clinical_score])
    # #             print(f'左手 ERD 與臨床分數 {clinical_score} 的相關性：{correlation_left}')

    # #             # 計算右手 ERD 與臨床分數的相關性
    # #             correlation_right, _ = pearsonr([right_mean[i]], [clinical_score])
    # #             print(f'右手 ERD 與臨床分數 {clinical_score} 的相關性：{correlation_right}')
    # # else:
    # #     print('左右手 ERD 數據的長度不一致')
           
    #     # 使用 open() 函數打開檔案，'w' 表示寫入模式
    # #     with open(file_path, 'a') as file:
    # #         file.write('right' +str(i) +':' +'\t'+str(round(right_average_of_negatives_value,2)) + "\t" +str(round(right_std_of_negatives_value,2))+ "\t"+str(round(right_max_negatives_value,2)) +"\n")
    # # # 提示消息
    # # print(f"資料已成功寫入 {file_path} 檔案。")

    #     # ERDS ratio
    #     # my_erds = ERDS(rest_start=rest_start, rest_end=rest_end, onset=onset)
    #     # erds    = my_erds.get_erds_ratio(data, freq_band=freq_band)

# %%
