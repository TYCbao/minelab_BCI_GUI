from tkinter import *
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import plot_map
import plot_erds
from my_decoder_FFT import fft_fig
from band_power import EEG_GUI

class result_page():
    def __init__(self, file_dict, sub_num):
        self.file_dict = file_dict
        self.sub_num = sub_num
        self.win_result = Tk()
        self.win_result.title("BCI for 榮總(Result)")
        self.win_result.minsize(width=1800, height=900)
        # self.win.state('zoomed')
        # self.page = Frame(self.win)
        # self.page.place(x=0, y=0) 

        self.label_1 = Label(self.win_result, text=f'Subject :{self.sub_num}', font=("Arial", 14, 'bold'),padx=5, pady=5)
        self.label_1.place(x=10, y=1) 
        
        label_x = 10
        for exp, file_path in self.file_dict.items():
            print(exp, file_path)
            Label(self.win_result, text=exp, font=("Arial", 14, 'bold'),padx=5, pady=5 ).place(x = label_x , y=50)
            
            if exp == 'Rest State':
                fig_map ,band_ratios = plot_map.plot_topology_map(file_path)
                canvas = FigureCanvasTkAgg(fig_map, master=self.win_result)
                canvas.draw()
                canvas.get_tk_widget().place(x=label_x, y=100)

                fft_file_path = file_path
                # 將字典轉換為字串
                band_ratio_text = " ".join([f"{band}: {ratio}" for band, ratio in band_ratios.items()])
                # print(band_ratio_text)
                Label(self.win_result, text=band_ratio_text, font=("Arial", 14, 'bold'),padx=5, pady=5).place(x = label_x , y=925)
                
                button_fft = Button(self.win_result, text="FFT", command=lambda: self.fft_gui(fft_file_path))
                button_fft.place(x=label_x, y=960)
                # 配置畫布大小
                canvas.get_tk_widget().config(width=550, height=820)

                button_power = Button(self.win_result, text="Band Power", command=lambda:EEG_GUI(fft_file_path))
                button_power.place(x=label_x+50, y=960)
                
                

            if exp == 'N-Back':
                # Label(self.win, text=exp, font=("Arial", 14, 'bold'),padx=5, pady=5 ).place(x = label_x , y=100)
                fig_map ,band_ratios= plot_map.plot_topology_map(file_path)
                canvas = FigureCanvasTkAgg(fig_map, master=self.win_result)
                canvas.draw()
                canvas.get_tk_widget().place(x=label_x, y=100)
                nback_file_path = file_path

                # 將字典轉換為字串
                band_ratio_text = " ".join([f"{band}: {ratio}" for band, ratio in band_ratios.items()])
                # print(band_ratio_text)
                Label(self.win_result, text=band_ratio_text, font=("Arial", 14, 'bold'),padx=5, pady=5).place(x = label_x , y=925)
                
                button_fft = Button(self.win_result, text="FFT", command=lambda: self.fft_gui(nback_file_path))
                button_fft.place(x=label_x, y=960)
                # 配置畫布大小
                canvas.get_tk_widget().config(width=550, height=820)

                button_power = Button(self.win_result, text="Band Power", command=lambda:EEG_GUI(nback_file_path))
                button_power.place(x=label_x+50, y=960)

                
            if exp == 'Motor Imagery':
                # Label(self.win, text=exp, font=("Arial", 14, 'bold'),padx=5, pady=5 ).place(x = label_x , y=100)
                fig_alpha, fig_beta = plot_erds.erd_value_with_file(file_path)
                canvas = FigureCanvasTkAgg(fig_alpha, master=self.win_result)
                canvas.draw()
                canvas.get_tk_widget().place(x=label_x, y=100)

                # 配置畫布大小
                canvas.get_tk_widget().config(width=450, height=820)

                canvas = FigureCanvasTkAgg(fig_beta, master=self.win_result)
                canvas.draw()
                canvas.get_tk_widget().place(x=label_x+400, y=100)

                # 配置畫布大小
                canvas.get_tk_widget().config(width=450, height=820)

            label_x += 550
            
        # 添加一個退出按鈕
        button_quit = Button(self.win_result, text="Quit", command=self.win_result.quit)
        button_quit.place(x=1800, y=950)
        self.win_result.mainloop()
    
    def fft_gui(self, path):
        fft_win = Toplevel(self.win_result)
        fft_win.title("FFT Window")
        fft_win.state('zoomed')  # 设置窗口大小

        fig_fft = fft_fig(path)
        canvas = FigureCanvasTkAgg(fig_fft, master=fft_win)
        canvas.draw()
        canvas.get_tk_widget().place(x=0, y=0)

        button_quit = Button(fft_win, text="Close FFT", command=fft_win.destroy)
        button_quit.place(x=1800, y=950)



if __name__ == "__main__":
    path = {'Rest State':'D:/GUI/rest_5/2024_05_20_1435/1.txt',
            'N-Back':'D:/GUI/h/2024_07_16_1714/1.txt',
            'Motor Imagery':'D:/GUI/240417030/2024_04_17_1442'}
    sub_num=1
    result_page(path, sub_num)
