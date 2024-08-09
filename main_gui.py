from tkinter import *
from tkinter import filedialog, messagebox
import mne
import os

from result_gui_test import result_page

def start_gui(root=None):
    if root is None:
        root = Tk()
    app = StartGUI(root)
    root.mainloop()

class StartGUI():
    def __init__(self, master) -> None:
        self.win1 = master
        self.win1.title("BCI for 榮總")
        self.win1.minsize(width=460, height=460)
        self.page = Frame(self.win1)
        self.page.place(x=0,y=0, width=460, height=460)
        self.label = Label(self.page, text='Subject Number:', font=("Arial", 14, 'bold'),padx=5, pady=5)
        self.label.place(x=20, y=20)

        self.entry_var = StringVar()
        self.entry = Entry(self.page, width=20,font=("Arial",14,"bold"),state=NORMAL,bg='white', textvariable=self.entry_var)
        self.entry.place(x=200, y=25)
            
        self.check_rest = BooleanVar()
        self.check_nback = BooleanVar()
        self.check_MI = BooleanVar()

        ###  按鈕設置  ###
        self.checkbutton1 = Checkbutton(self.page, text = 'Rest State', font=("Arial",14,"bold"),variable=self.check_rest)
        self.checkbutton1.place(x=20, y=80)
        self.checkbutton2 = Checkbutton(self.page, text = 'N-Back ', font=("Arial",14,"bold"),variable=self.check_nback)
        self.checkbutton2.place(x=20, y=180)
        self.checkbutton3 = Checkbutton(self.page, text = 'Motor Imagery', font=("Arial",14,"bold"),variable=self.check_MI)
        self.checkbutton3.place(x=20, y=280)

        ###  功能介紹  ###
        self.label_RS1 = Label(self.page, text='1. Plot a brain topology map ', font=("Arial", 10, 'bold'))
        self.label_RS1.place(x=80, y=110)
        self.label_RS2 = Label(self.page, text='2. Display values of F3, F4, C3, and C4', font=("Arial", 10, 'bold'))
        self.label_RS2.place(x=80, y=135)

        self.label_nb1 = Label(self.page, text='1. Plot a brain topology map ', font=("Arial", 10, 'bold'))
        self.label_nb1.place(x=80, y=210)
        self.label_nb2 = Label(self.page, text='2. Display values of F3, F4, C3, and C4', font=("Arial", 10, 'bold'))
        self.label_nb2.place(x=80, y=235)

        self.label_mi1 = Label(self.page, text='1. Display the ERD/ERS values of motor imagery ', font=("Arial", 10, 'bold'))
        self.label_mi1.place(x=80, y=310)

        self.label_2 = Label(self.page, text='Designed by YuChing Tai (NCUEE-Minelab)', font=("Arial", 8, 'bold'), fg= '#6C6C6C' ,padx=5, pady=5)
        self.label_2.place(x=220, y=430)

        self.button = Button(self.page, text = 'NEXT',font=("Arial", 10, 'bold'),command = self.show)
        self.button.place(x=380, y=400)
        
    def show(self):
        print('受試者編號:',self.entry.get())
        print('Rest:', self.check_rest.get() )
        print('N-back:', self.check_nback.get())
        print('Motor Imagery:',self.check_MI.get())
        sub_num = self.entry.get()
        rest = self.check_rest.get()
        nback = self.check_nback.get()
        mi = self.check_MI.get()

        second_page(self.win1, sub_num, rest, nback, mi)
        self.page.destroy()
        
        

class second_page():
    def __init__(self, master, sub_num, rest, nback, mi):
        self.win2 = master
        self.sub_num = sub_num

        self.win2.title("BCI for 榮總(choose file)")
        self.win2.minsize(width=460, height=460)
        self.page = Frame(self.win2) 
        self.page.place(x=0,y=0, width=460, height=460)

        self.label_1 = Label(self.page, text=f'Choose File ', font=("Arial", 14, 'bold'),padx=5, pady=5)
        self.label_1.place(x=20, y=20)

        self.label_2 = Label(self.page, text=f'Subject :{self.sub_num}', font=("Arial", 12, 'bold'),padx=5, pady=5)
        self.label_2.place(x=20, y=60)

        self.label_3 = Label(self.page, text='Designed by YuChing Tai (NCUEE-Minelab)', font=("Arial", 8, 'bold'), fg= '#6C6C6C', padx=5, pady=5)
        self.label_3.place(x=220, y=430)
   
        self.file_paths = {}
        y_label=100
        if rest:
            self.file_input("Rest State File:", y_label, "Rest State")
            y_label+=50
        if nback:
            self.file_input("N-back File:", y_label, "N-Back")
            y_label+=50
        if mi:
            self.file_input("Motor Imagery:", y_label, "Motor Imagery")
            y_label+=50

        self.next_button = Button(self.page, text = 'NEXT',font=("Arial", 10, 'bold'),command=self.show_next)
        self.next_button.place(x=380, y=400)

        self.up_button = Button(self.page, text = 'BACK',font=("Arial", 10, 'bold'),command=self.go_back)
        self.up_button.place(x=320, y=400)

        # Call check_next_button_state once during initialization
        self.check_next_button_state()

    def file_input(self, label_text, y_label, key):
        label = Label(self.page, text=f'{label_text}', font=("Arial", 10, 'bold'),padx=5, pady=5)
        label.place(x=20, y=y_label)

        file_var = StringVar()
        self.file_paths[key] = file_var

       
        file_entry = Entry(self.page, width=30, font=("Arial", 10), state=NORMAL, textvariable=file_var)
        file_entry.place(x=140, y=y_label+5)
        
        file_button = Button(self.page, text="Browse", font=("Arial", 10), command=lambda:self.browse_file(key, file_var), bg='#C7C7E2')
        file_button.place(x=360, y=y_label)

    def browse_file(self, key, file_var):
        if key=='Motor Imagery':
            while True:
                file_path = filedialog.askdirectory()
                if file_path:
                    if self.check_required_files(file_path):
                        file_var.set(file_path)
                        break
                    else:
                        messagebox.showerror("Error", "Selected folder does not contain required files (1.txt and tri_1.txt). Please select another folder.")
        else:
            file_path = filedialog.askopenfilename()
            if file_path:
                file_var.set(file_path)
        self.check_next_button_state(file_path)

    def check_required_files(self, folder_path):
        required_files = ['1.txt', 'tri_1.txt']
        return all(os.path.isfile(os.path.join(folder_path, file)) for file in required_files)
    
    def check_next_button_state(self, *arg):
        # Check if all necessary files are selected
        all_files_selected = all(file_var.get() != '' for file_var in self.file_paths.values())
        if all_files_selected:
            self.next_button.config(state=NORMAL)
        else:
            self.next_button.config(state=DISABLED)

    def show_next(self):
        file_path_dict = {k: v.get() for k, v in self.file_paths.items()}
        print(file_path_dict)
        self.win2.destroy()
        result_page(file_path_dict, self.sub_num)

    def go_back(self):
        self.win2.destroy()
        win = Tk()
        StartGUI(win)




if __name__ == "__main__":
    win=Tk()
    StartGUI(win)
    win.mainloop()
    

    



    

