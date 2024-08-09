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
        self.win = Tk()
        self.win.title("BCI for 榮總(Result)")
        self.win.state('zoomed')

        self.win.bind("<Configure>", self.on_resize)

        self.label_1 = Label(self.win, text=f'Subject :{self.sub_num}', font=("Arial", 14, 'bold'), padx=5, pady=5)
        self.label_1.place(relx=0.01, rely=0.01)

        self.canvases = []
        self.labels = []
        self.buttons = []
        label_x = 0.01
        label_y = 0.05

        for exp, file_path in self.file_dict.items():
            Label(self.win, text=exp, font=("Arial", 14, 'bold'), padx=5, pady=5).place(relx=label_x, rely=label_y)
            
            if exp == 'Rest State':
                fig_map, band_ratios = plot_map.plot_topology_map(file_path)
                canvas = FigureCanvasTkAgg(fig_map, master=self.win)
                canvas.draw()
                canvas_widget = canvas.get_tk_widget()
                canvas_widget.place(relx=label_x, rely=label_y + 0.05, relwidth=0.5, relheight=0.8)
                self.canvases.append(canvas_widget)

                rest_file_path = file_path

                band_ratio_text = " ".join([f"{band}: {ratio}" for band, ratio in band_ratios.items()])
                band_label = Label(self.win, text=band_ratio_text, font=("Arial", 13, 'bold'), padx=5, pady=5)
                band_label.place(relx=label_x, rely=0.87)
                self.labels.append(band_label)

                button_fft = Button(self.win, text="FFT", command=lambda: self.fft_gui(rest_file_path))
                button_fft.place(relx=label_x, rely=0.92)
                self.buttons.append(button_fft)

                button_power = Button(self.win, text="Band Power", command=lambda: EEG_GUI(rest_file_path))
                button_power.place(relx=label_x + 0.02, rely=0.92)
                self.buttons.append(button_power)

            elif exp == 'N-Back':
                fig_map, band_ratios = plot_map.plot_topology_map(file_path)
                canvas = FigureCanvasTkAgg(fig_map, master=self.win)
                canvas.draw()
                canvas_widget = canvas.get_tk_widget()
                canvas_widget.place(relx=label_x, rely=label_y + 0.05, relwidth=0.5, relheight=0.8)
                self.canvases.append(canvas_widget)

                nback_file_path = file_path

                band_ratio_text = " ".join([f"{band}: {ratio}" for band, ratio in band_ratios.items()])
                band_label = Label(self.win, text=band_ratio_text, font=("Arial", 14, 'bold'), padx=5, pady=5)
                band_label.place(relx=label_x, rely=0.87)
                self.labels.append(band_label)

                button_fft = Button(self.win, text="FFT", command=lambda: self.fft_gui(nback_file_path))
                button_fft.place(relx=label_x, rely=0.92)
                self.buttons.append(button_fft)

                button_power = Button(self.win, text="Band Power", command=lambda: EEG_GUI(nback_file_path))
                button_power.place(relx=label_x + 0.02, rely=0.92)
                self.buttons.append(button_power)

            elif exp == 'Motor Imagery':
                fig_alpha, fig_beta = plot_erds.erd_value_with_file(file_path)
                canvas_alpha = FigureCanvasTkAgg(fig_alpha, master=self.win)
                canvas_alpha.draw()
                canvas_alpha_widget = canvas_alpha.get_tk_widget()
                canvas_alpha_widget.place(relx=label_x, rely=label_y + 0.05, relwidth=0.2, relheight=0.8)
                canvas_alpha_widget.config(width=200)
                self.canvases.append(canvas_alpha_widget)

                mi_file_path = file_path

                canvas_beta = FigureCanvasTkAgg(fig_beta, master=self.win)
                canvas_beta.draw()
                canvas_beta_widget = canvas_beta.get_tk_widget()
                canvas_beta_widget.place(relx=label_x + 0.23, rely=label_y + 0.05, relwidth=0.2, relheight=0.8)
                canvas_beta_widget.config(width=200)
                self.canvases.append(canvas_beta_widget)

                button_erd_alpha = Button(self.win, text="Alpha ERD/ERS", command=lambda: self.erd_gui(mi_file_path, band='Alpha'))
                button_erd_alpha.place(relx=label_x, rely=0.92)
                self.buttons.append(button_erd_alpha)

                button_erd_beta = Button(self.win, text="Beta ERD/ERS", command=lambda: self.erd_gui(mi_file_path, band='Beta'))
                button_erd_beta.place(relx=label_x+0.23, rely=0.92)
                self.buttons.append(button_erd_beta)

            label_x += 0.26

        button_quit = Button(self.win, text="Quit", command=self.win.quit)
        button_quit.place(relx=0.95, rely=0.96)

        button_home = Button(self.win, text="Home", command=self.back_home)
        button_home.place(relx=0.92, rely=0.96)

        self.label_2 = Label(self.win, text='Designed by YuChing Tai (NCUEE-Minelab)', font=("Arial", 8, 'bold'), fg= '#6C6C6C', padx=5, pady=5)
        self.label_2.place(relx=0.82, rely=0.01)


        self.win.mainloop()
    
    def on_resize(self, event):
        for canvas in self.canvases:
            canvas.place_configure(relwidth=0.25, relheight=0.8)
        for label in self.labels:
            label.place_configure(rely=0.92)
        for button in self.buttons:
            button.place_configure(rely=0.96)
    

    def fft_gui(self, path):
        fft_win = Toplevel(self.win)
        fft_win.title("FFT Window")
        fft_win.state('zoomed')

        fig_fft = fft_fig(path)
        canvas = FigureCanvasTkAgg(fig_fft, master=fft_win)
        canvas.draw()
        canvas.get_tk_widget().place(relx=0, rely=0, relwidth=1, relheight=1)

        button_quit = Button(fft_win, text="Close FFT", command=fft_win.destroy)
        button_quit.place(relx=0.95, rely=0.94)
    
    def erd_gui(self, path, band):
        erd_win = Toplevel(self.win)
        erd_win.title("ERD/ERS Window")
        erd_win.state('zoomed')

        erd_fig = plot_erds.create_erd_image(path, band)
        canvas = FigureCanvasTkAgg(erd_fig, master=erd_win)
        canvas.draw()
        canvas.get_tk_widget().place(relx=0, rely=0, relwidth=1, relheight=1)

        button_quit = Button(erd_win, text="Close ERD/ERS", command=erd_win.destroy)
        button_quit.place(relx=0.93, rely=0.96)
    
    def back_home(self):
        self.win.destroy()
        from main_gui import start_gui
        root = Tk()
        start_gui(root)




if __name__ == "__main__":
    path = {'Rest State':'D:/GUI/rest_5/2024_05_20_1435/1.txt',
            'N-Back':'D:/GUI/h/2024_07_16_1714/1.txt',
            'Motor Imagery':'D:/GUI/240417030/2024_04_17_1442'}
    sub_num=1
    result_page(path, sub_num)
