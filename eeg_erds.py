#%%
import math
import scipy.io
import numpy as np
from eeg_decoder import Filter
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
# %config InlineBackend.figure_format = 'retina'

#%%
class ERDS():
    def __init__(self, rest_start=0, rest_end=2, onset=2, fs=1000):
        self.fs = fs
        self.rest_start = int(self.fs * rest_start) # time for starting rest, in second
        self.rest_end   = int(self.fs * rest_end) # time for ending rest, in second, 從休息開始到結束的長度，動作前 2 秒當作休息
        self.onset      = int(self.fs * onset) # time for action onset, in second
        self.smooth_factor = 50 # for smooth window size which = len(eeg_data)/smooth_factor, the larger the value, the smaller the window size
        self.freq_bands = {
            'delta' : [1,   4],
            'theta' : [4,   8],
            'alpha' : [8,  12],
            'beta'  : [12, 30],
            'gamma' : [30, 50]}
        self.filter = Filter()
    
    def get_erds(self, eeg_data, freq_band='alpha'):
        '''
        code refer to : https://blog.csdn.net/qq_40166660/article/details/110084982
        1. 帶通濾波，濾出alpha band或其他頻段的訊號
        2. 對所有訊號取平方獲得訊號能量
        3. 對試驗取平均
        4. 平滑處理
        5. 取休息的平均，拿動作前2秒當作休息，並計算ERD/ERS

        Parameters 
        ----------------------------------------
        `eeg_data` : same class eeg data (e.g. left hand class) with shape = (trials, channels, samples) which
            trials : number of trials
            channels : number of channel
            samples : sample points of each trial
            
        `band` : frequency band for band pass filter, 
            'delta' :  1 -   4 Hz
            'theta' :  4 -   8 Hz
            'alpha' :  8 -  12 Hz
            'beta'  : 12 -  30 Hz
            'gamma' : 30 - 100 Hz

            for custom value, input two values in list format. e.g. [10, 20]

        Return
        --------------------------------------------------
        `erds` : ERD/ERS of eeg_data, shape = (num_channel, samples)

        Examples
        --------------------------------------------------
        >>> my_erds = ERDS(rest_start=0, rest_end=2, onset=2)
        >>> erds = my_erds.get_erds(eeg_data, freq_band='alpha')        
        '''
                
        # 1. Bandpass filtering    
        if isinstance(freq_band, list):
            self.low_cut = freq_band[0]
            self.high_cut = freq_band[1]      
        else:     
            self.low_cut = self.freq_bands[freq_band][0]
            self.high_cut = self.freq_bands[freq_band][1]       

        print(self.low_cut, self.high_cut) 
        # print('eeg_data.shape',eeg_data.shape)
        bandpass_data = np.zeros(eeg_data.shape)
        for i in range(eeg_data.shape[1]): # for each channel
            bandpass_data[:, i] = self.filter.butter_bandpass_filter(eeg_data[:, i], self.low_cut, self.high_cut, self.fs) 

        # 2. Squaring
        power_data = np.power(bandpass_data, 2)

        # 3. Average all trials 
        avg_power = np.average(power_data, axis=0) # average all trials, resulting shape = (num_channel, samples)

        # 4. Smooth
        # for i in range(avg_power.shape[0]): # for each channel
        #     avg_power[i] = savgol_filter(avg_power[i], len(avg_power[i])//self.smooth_factor, 3) # smooth the data, window size = len(data)//smooth_factor, polynomial order 3
        SMOOTH=200
        new_cut_data = np.zeros((19,8000-SMOOTH+1))
        for i in range(19):
            new_cut_data[i] = np.convolve(avg_power[i], np.ones(SMOOTH), 'valid') / SMOOTH
        # 5. Calculate ERD/ERS 
        # average rest power
        data_rest = new_cut_data.T[self.rest_start : self.rest_end].T # (3, n).T -> (n, 3) -> (k, 3).T -> (3, k)  
        avg_rest = np.average(data_rest, axis = 1) # average rest of each channel, shape = (num_channel, samples)

        # ERDS = (avg_power - rest_avg_power) / rest_avg_power (%)
        erds = np.zeros(new_cut_data.shape)
        for i in range(erds.shape[0]): # for each channel
            for j in range(erds.shape[1]): # for each samples
                erds[i, j] = ((new_cut_data[i, j] - avg_rest[i]) / avg_rest[i])*100

        return erds


    def get_erds_ratio(self, eeg_data, freq_band='alpha'):
        '''
        code refer to : https://blog.csdn.net/qq_40166660/article/details/110084982
        1. 帶通濾波，濾出alpha band或其他頻段的訊號
        2. 對所有訊號取平方獲得訊號能量
        3. 對試驗取平均
        4. 平滑處理
        5. 取休息的平均，拿動作前2秒當作休息，
        6. 去動作的平均，並計算ERD/ERS

        Parameters 
        ----------------------------------------
        `eeg_data` : same class eeg data (e.g. left hand class) with shape = (trials, channels, samples) which
            trials : number of trials
            channels : number of channel
            samples : sample points of each trial
            
        `band` : frequency band for band pass filter, 
            'delta' :  1 -   4 Hz
            'theta' :  4 -   8 Hz
            'alpha' :  8 -  12 Hz
            'beta'  : 12 -  30 Hz
            'gamma' : 30 - 100 Hz

            for custom value, input two values in list format. e.g. [10, 20]

        Return
        --------------------------------------------------
        `erds` : ERD/ERS of eeg_data, shape = (num_channel, samples)

        Examples
        --------------------------------------------------
        >>> my_erds = ERDS(rest_start=0, rest_end=2, onset=2)
        >>> erds = my_erds.get_erds(eeg_data, freq_band='alpha')        
        '''
                
        # 1. Bandpass filtering    
        if isinstance(freq_band, list):
            self.low_cut = freq_band[0]
            self.high_cut = freq_band[1]      
        else:     
            self.low_cut = self.freq_bands[freq_band][0]
            self.high_cut = self.freq_bands[freq_band][1]
        bandpass_data = np.zeros(eeg_data.shape)
        for i in range(eeg_data.shape[1]): # for each channel
            bandpass_data[:, i] = self.filter.butter_bandpass_filter(eeg_data[:, i], self.low_cut, self.high_cut, self.fs) 

        # 2. Squaring
        power_data = np.power(bandpass_data, 2)

        # 3. Average all trials 
        avg_power = np.average(power_data, axis=0) # average all trials, resulting shape = (num_channel, samples)

        # 4. Smooth
        for i in range(avg_power.shape[0]): # for each channel
            avg_power[i] = savgol_filter(avg_power[i], len(avg_power[i])//self.smooth_factor, 3) # smooth the data, window size = len(data)//smooth_factor, polynomial order 3

        # 5. Calculate ERD/ERS 
        # average rest power
        data_rest = avg_power.T[self.rest_start : self.rest_end].T # (3, n).T -> (n, 3) -> (k, 3).T -> (3, k)  
        avg_rest = np.average(data_rest, axis = 1) # average rest of each channel, shape = (num_channel, samples)

        # average mi power, 只有模型判斷到的區間
        data_move = avg_power.T[self.onset : self.onset + 3000].T
        avg_move = np.average(data_move, axis = 1)

        # ERDS = (avg_power - rest_avg_power) / rest_avg_power (%)
        erds_ratio = ((avg_move - avg_rest)/avg_rest)*100

        return erds_ratio


    ### 8 channels ###
    def plot_erds(self, erds_data, save_dir = '.', title="Hands ERD/ERS", channel_list=['F3', 'F4', 'C3', 'Cz', 'C4', 'P3', 'Pz', 'P4']):
        """
        plot calculated ERD/ERS value

        Parameters
        ----------------------------------------
        `erds_data` : ERD/ERS of eeg_data with shape = (num_channel, samples) which,
                      num_channel is the number of channel,
                      samples is the sample points of each channel  
                     
        `title` : fig title
        
        `channel_list` : list of channel names, e.g. ['C3', 'C4']


        Examples
        ----------------------------------------
        >>> my_erd = ERDS(rest_start=0, rest_end=3, onset=3, fs=1000) # 取動作前三秒當作休息，第三秒動作開始
        >>> erds = my_erd.get_erds(left_hand, freq_band='alpha')
        >>> my_erd.plot_erds(erds, title="Left Hand ERD/ERS", channel_list=['C3', 'C4'])
                
        """   

        channel_loc = {'F3': (0, 0), 'Fz': (0, 1), 'F4': (0, 2), 
                       'C3': (1, 0), 'Cz': (1, 1), 'C4': (1, 2), 
                       'P3': (2, 0), 'Pz': (2, 1), 'P4': (2, 2)}
        

        # plt.style.use('ggplot')
        plt.style.use('_classic_test_patch')        
        xticks = np.arange(0, len(erds_data[0])+1, self.fs)
        xticklabels = [str(i-(self.onset/self.fs)) for i in range(len(xticks))]


        xaxis = np.arange(erds_data.shape[1])

        fig, ax = plt.subplots(3, 3, figsize=(10, 6))
        # disable the subplots that not show in channel_list
        chs = list(channel_loc.keys())
        for ch in channel_list:
            if ch in channel_loc.keys():
                chs.remove(ch)
        for ch in chs:
            ax[channel_loc[ch]].remove()

        # plot ERD/ERS
        for i, ch in enumerate(channel_list): # for each channel
            row = channel_loc[ch][0]
            col = channel_loc[ch][1]                            

            # ax[row, col].plot(xaxis, erds_data[i], label=channel_list[i], c='dimgray')
            ax[row, col].plot(xaxis, erds_data[i], label=channel_list[i], c='black', lw = 0.4)
            ax[row, col].plot([self.onset, self.onset], [np.min(erds_data), np.max(erds_data)], '--', c='gray', alpha=0.8) # onset line
            ax[row, col].plot([0, erds_data.shape[1]+1], [0, 0], '--', c='black') # baseline (0)
            ax[row, col].set_xticks(xticks)
            ax[row, col].set_xticklabels(xticklabels)

            # fill the color of ERD/ERS regions
            baseline = np.zeros((len(erds_data[i])))
            ers_region = np.zeros((len(baseline)))
            erd_region = np.zeros((len(baseline)))

            rest_region = np.zeros((len(baseline)))

            for j in range(self.onset, len(baseline)):
                # ERD region
                if(erds_data[i][j] <= baseline[j]):
                    erd_region[j] = True
                else: 
                    erd_region[j] = False

                # ERS region                
                if(erds_data[i][j] > baseline[j]):
                    ers_region[j] = True
                else:
                    ers_region[j] = False
                
            # rest region
            # rest_region[self.rest_start: self.rest_end] = True
            # ax[i].fill_between(xaxis, baseline, erds_data[i], where=rest_region, facecolor='gray', interpolate=True, alpha=0.3) # brown                              
                           
            ax[row, col].fill_between(xaxis, baseline, erds_data[i], where=erd_region, facecolor='#17395C', interpolate=True, alpha=0.9)
            ax[row, col].fill_between(xaxis, baseline, erds_data[i], where=ers_region, facecolor='firebrick', interpolate=True, alpha=0.9) # brown                              
            ax[row, col].set_xlabel("Time (sec)", fontsize=8)
            ax[row, col].legend(loc='lower right', fontsize='medium', labelcolor='#F25C05')
            ax[row, col].set_ylabel("%", fontsize=10)

            ax[row, col].spines[['right', 'top']].set_visible(False)
            ax[row, col].set_ylim([-100, 100])            


        freq_annotation = f'[{self.low_cut}, {self.high_cut}] Hz'
        fig.suptitle(title + '\n' + freq_annotation, fontsize=14)
        fig.tight_layout()

        plt.savefig(f"{save_dir}/{' '.join(title.split(' ')[0:2]).title()}.png", dpi = 500)
        # plt.show()

    ### 19 channels ###
    def plot_erds_19(self, erds_data, save_dir = '.', title="Hands ERD/ERS", channel_list=['Fp1','Fp2','F7','F3','Fz', 'F4','F8','T3', 'C3', 'Cz', 'C4','T4','T5', 'P3', 'Pz', 'P4','T6','O1','O2']):
        """
        plot calculated ERD/ERS value

        Parameters
        ----------------------------------------
        `erds_data` : ERD/ERS of eeg_data with shape = (num_channel, samples) which,
                      num_channel is the number of channel,
                      samples is the sample points of each channel  
                     
        `title` : fig title
        
        `channel_list` : list of channel names, e.g. ['C3', 'C4']


        Examples
        ----------------------------------------
        >>> my_erd = ERDS(rest_start=0, rest_end=3, onset=3, fs=1000) # 取動作前三秒當作休息，第三秒動作開始
        >>> erds = my_erd.get_erds(left_hand, freq_band='alpha')
        >>> my_erd.plot_erds(erds, title="Left Hand ERD/ERS", channel_list=['C3', 'C4'])
                
        """   

        channel_loc = {            'Fp1':(0, 1),               'Fp2':(0, 3),
                       'F7':(1, 0),'F3': (1, 1), 'Fz': (1, 2), 'F4': (1, 3), 'F8':(1, 4), 
                       'T3':(2, 0),'C3': (2, 1), 'Cz': (2, 2), 'C4': (2, 3), 'T4':(2, 4),
                       'T5':(3, 0),'P3': (3, 1), 'Pz': (3, 2), 'P4': (3, 3), 'T6':(3, 4),
                                   'O1':(4, 1),                'O2':(4, 3)}
        

        # plt.style.use('ggplot')
        plt.style.use('_classic_test_patch')        
        xticks = np.arange(0, len(erds_data[0])+1, self.fs)
        xticklabels = [str(i-(self.onset/self.fs)) for i in range(len(xticks))]


        xaxis = np.arange(erds_data.shape[1])

        fig, ax = plt.subplots(5, 5, figsize=(15, 10))
        # disable the subplots that not show in channel_list
        chs = list(channel_loc.keys())
        for ch in channel_list:
            if ch in channel_loc.keys():
                chs.remove(ch)
        for ch in chs:
            ax[channel_loc[ch]].remove()

        # plot ERD/ERS
        for i, ch in enumerate(channel_list): # for each channel
            row = channel_loc[ch][0]
            col = channel_loc[ch][1]                            

            # ax[row, col].plot(xaxis, erds_data[i], label=channel_list[i], c='dimgray')
            ax[row, col].plot(xaxis, erds_data[i], label=channel_list[i], c='black', lw = 0.4)
            ax[row, col].plot([self.onset, self.onset], [np.min(erds_data), np.max(erds_data)], '--', c='gray', alpha=0.8) # onset line
            ax[row, col].plot([0, erds_data.shape[1]+1], [0, 0], '--', c='black') # baseline (0)
            ax[row, col].set_xticks(xticks)
            ax[row, col].set_xticklabels(xticklabels)

            # fill the color of ERD/ERS regions
            baseline = np.zeros((len(erds_data[i])))
            ers_region = np.zeros((len(baseline)))
            erd_region = np.zeros((len(baseline)))

            rest_region = np.zeros((len(baseline)))

            for j in range(self.onset, len(baseline)):
                # ERD region
                if(erds_data[i][j] <= baseline[j]):
                    erd_region[j] = True
                else: 
                    erd_region[j] = False

                # ERS region                
                if(erds_data[i][j] > baseline[j]):
                    ers_region[j] = True
                else:
                    ers_region[j] = False
                
            # rest region
            # rest_region[self.rest_start: self.rest_end] = True
            # ax[i].fill_between(xaxis, baseline, erds_data[i], where=rest_region, facecolor='gray', interpolate=True, alpha=0.3) # brown                              
                           
            ax[row, col].fill_between(xaxis, baseline, erds_data[i], where=erd_region, facecolor='#17395C', interpolate=True, alpha=0.9)
            ax[row, col].fill_between(xaxis, baseline, erds_data[i], where=ers_region, facecolor='firebrick', interpolate=True, alpha=0.9) # brown                              
            ax[row, col].set_xlabel("Time (sec)", fontsize=8)
            ax[row, col].legend(loc='lower right', fontsize='medium', labelcolor='#F25C05')
            ax[row, col].set_ylabel("%", fontsize=10)

            ax[row, col].spines[['right', 'top']].set_visible(False)
            ax[row, col].set_ylim([-100, 100])            

        fig.text( 0.03, 0.965,'Number: '+ save_dir[-25:-16] + 'Date: ' + save_dir[-15:], fontsize=14)
        freq_annotation = f'[{self.low_cut}, {self.high_cut}] Hz'
        fig.suptitle(title + '\n' + freq_annotation, fontsize=14)
        fig.tight_layout()

        plt.savefig(f"{save_dir}/{' '.join(title.split(' ')[0:2]).title()}.png", dpi = 500)
        
        return fig

    # 只有C3，Cz和C4
    # def plot_erds(self, erds_data, save_dir = '.', title="Hands ERD/ERS",channel_list=['C3', 'Cz', 'C4']):
    #     """
    #     plot calculated ERD/ERS value

    #     Parameters
    #     ----------------------------------------
    #     `erds_data` : ERD/ERS of eeg_data with shape = (num_channel, samples) which,
    #                   num_channel is the number of channel,
    #                   samples is the sample points of each channel  
                     
    #     `title` : fig title
        
    #     `channel_list` : list of channel names, e.g. ['C3', 'C4']


    #     Examples
    #     ----------------------------------------
    #     >>> my_erd = ERDS(rest_start=0, rest_end=3, onset=3, fs=1000) # 取動作前三秒當作休息，第三秒動作開始
    #     >>> erds = my_erd.get_erds(left_hand, freq_band='alpha')
    #     >>> my_erd.plot_erds(erds, title="Left Hand ERD/ERS", channel_list=['C3', 'C4'])
                
    #     """        

    #     # plt.style.use('ggplot')
    #     plt.style.use('_classic_test_patch')
    #     xticks = np.arange(0, len(erds_data[0])+1, self.fs)
    #     xticklabels = [str(i-(self.onset/self.fs)) for i in range(len(xticks))]


    #     xaxis = np.arange(erds_data.shape[1])

    #     fig, ax = plt.subplots(1, 3, figsize=(10, 2))
    #     # disable the subplots that not show in channel_list


    #     # plot ERD/ERS
    #     for i, ch in enumerate(channel_list): # for each channel                        
    #         # ax[row, col].plot(xaxis, erds_data[i], label=channel_list[i], c='dimgray')
    #         ax[i].plot(xaxis, erds_data[i], label=channel_list[i], c='black', lw = 0.4)
    #         ax[i].plot([self.onset, self.onset], [np.min(erds_data), np.max(erds_data)], '--', c='gray', alpha=0.8) # onset line
    #         ax[i].plot([0, erds_data.shape[1]+1], [0, 0], '--', c='black') # baseline (0)
    #         ax[i].set_xticks(xticks)
    #         ax[i].set_xticklabels(xticklabels)

    #         # fill the color of ERD/ERS regions
    #         baseline = np.zeros((len(erds_data[i])))
    #         ers_region = np.zeros((len(baseline)))
    #         erd_region = np.zeros((len(baseline)))

    #         rest_region = np.zeros((len(baseline)))

    #         for j in range(self.onset, len(baseline)):
    #             # ERD region
    #             if(erds_data[i][j] <= baseline[j]):
    #                 erd_region[j] = True
    #             else: 
    #                 erd_region[j] = False

    #             # ERS region                
    #             if(erds_data[i][j] > baseline[j]):
    #                 ers_region[j] = True
    #             else:
    #                 ers_region[j] = False
                
    #         # rest region
    #         # rest_region[self.rest_start: self.rest_end] = True
    #         # ax[i].fill_between(xaxis, baseline, erds_data[i], where=rest_region, facecolor='gray', interpolate=True, alpha=0.3) # brown                              
                           
    #         ax[i].fill_between(xaxis, baseline, erds_data[i], where=erd_region, facecolor='#17395C', interpolate=True, alpha=0.9)
    #         ax[i].fill_between(xaxis, baseline, erds_data[i], where=ers_region, facecolor='firebrick', interpolate=True, alpha=0.9) # brown                              
    #         ax[i].set_xlabel("Time (sec)", fontsize=8)
    #         # ax[i].legend(loc='lower right', fontsize='medium', labelcolor='#F25C05')
    #         ax[i].set_ylabel("%", fontsize=10)
    #         ax[i].spines[['right', 'top']].set_visible(False)

    #         ax[i].set_ylim([-100, 100])

    #     freq_annotation = f'[{self.low_cut}, {self.high_cut}] Hz'
    #     # fig.suptitle(title, fontsize=14)
    #     fig.tight_layout()

    #     plt.savefig(f"{save_dir}/{' '.join(title.s  plit(' ')[0:2]).title()}.png", dpi = 500)
        # plt.show()    

if __name__=="__main__":
    # Dataset : https://www.bbci.de/competition/ii/
    # PATH = r"D:\user\Desktop\minelab\EEG_data_recv\EEG_DATA_1.txt"
    mat = scipy.io.loadmat('dataset_BCIcomp1.mat')
    # mat = scipy.io.loadmat(PATH)

    x_train = mat['x_train']
    x_test = mat['x_test']
    y_train = mat['y_train']

    left_hand = []
    right_hand = []

    x_train = np.transpose(x_train, (2, 1, 0))

    for i in range(len(y_train)):
        if y_train[i] == 1:
            left_hand.append(x_train[i])
        elif y_train[i] == 2:
            right_hand.append(x_train[i])

    left_hand = np.array(left_hand)
    right_hand = np.array(right_hand)
    print(left_hand.shape)
    print(right_hand.shape)

    aaa = []
    left_hand = np.transpose(left_hand, (1,0,2))
    aaa.append(left_hand[0])
    aaa.append(left_hand[2])
    left_hand = np.transpose(aaa, (1,0,2))

    aaa = []
    right_hand = np.transpose(right_hand, (1,0,2))
    aaa.append(right_hand[0])
    aaa.append(right_hand[2])
    right_hand = np.transpose(aaa, (1,0,2))

    my_erd = ERDS(rest_start=1, rest_end=2, onset=3, fs=128)
    erds = my_erd.get_erds(left_hand, freq_band='alpha')
    my_erd.plot_erds(erds, title="Left Hand ERD/ERS")
    erds = my_erd.get_erds(right_hand, freq_band='alpha') # [8, 14]
    my_erd.plot_erds(erds, title="Right Hand ERD/ERS")
    plt.show()
    # -------------------------------------------------------#

    
    # x_train = np.load('./trials_seg.npy')

    # aaa = []
    # left_hand = np.transpose(x_train, (1,0,2))
    # aaa.append(left_hand[2])
    # aaa.append(left_hand[4])
    # left_hand = np.transpose(aaa, (1,0,2))

    # my_erd = ERDS(rest_start=0, rest_end=2, onset=2, fs=1000) 
    # erds = my_erd.get_erds(left_hand, freq_band='delta')
    # my_erd.plot_erds(erds, title="Left Hand ERD/ERS", channel_list=['C3', 'C4'])
    # plt.show()

