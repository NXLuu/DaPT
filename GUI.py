import tkinter
import tkinter.messagebox
import customtkinter
import tkinter

from pandas._libs.missing import maxsize

import classifier
from tkinter.ttk import Progressbar
import customtkinter
import pygame
from PIL import Image, ImageTk
from threading import *
import time
import math
from tkinter  import Tk     # from tkinter import Tk for Python 3.x
from tkinter.filedialog import askopenfilename
import matplotlib.pyplot as plt
import  scipy
from scipy.io import wavfile as wav
from scipy.fftpack import fft
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('TkAgg')

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg,
    NavigationToolbar2Tk
)

from pydub import AudioSegment


customtkinter.set_appearance_mode("System")  # Modes: "System" (standard), "Dark", "Light"
customtkinter.set_default_color_theme("blue")  # Themes: "blue" (standard), "green", "dark-blue"


pygame.mixer.init()



class App(customtkinter.CTk):

    WIDTH =  0
    HEIGHT = 0

    def __init__(self):
        super().__init__()
        self.pack_propagate(0)
        screen_width = str(self.winfo_screenwidth())
        screen_height = str(self.winfo_screenheight())
        self.geometry(screen_width + "x" + screen_height + "+0+0")
        customtkinter.set_appearance_mode("dark")
        self.title("Nhận diện âm thanh nhạc cụ")
        # self.geometry(f"{WIDTH}x{HEIGHT}")
        # self.attributes('-fullscreen', True)
        self.protocol("WM_DELETE_WINDOW", self.on_closing)  # call .on_closing() when app gets closed
        self.file_name = '';
        self.is_play = False;

        # ============ create two frames ============

        # configure grid layout (2x1)
        self.grid_columnconfigure(2, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.frame_left = customtkinter.CTkFrame(master=self,
                                                width=500,
                                                 corner_radius=0)
        self.frame_left.grid(row=0, column=1, sticky="nswe")

        self.frame_right = customtkinter.CTkFrame(master=self)
        self.frame_right.grid(row=0, column=2, sticky="nswe", padx=20, pady=20)

        # ============ frame_left ============

        # configure grid layout (1x11)


        self.frame_left.grid_rowconfigure(0, minsize=10)   # empty row with minsize as spacing
        self.frame_left.grid_rowconfigure(5, weight=1)  # empty row as spacing
        self.frame_left.grid_rowconfigure(8, minsize=20)    # empty row with minsize as spacing
        self.frame_left.grid_rowconfigure(11, minsize=10)  # empty row with minsize as spacing

        self.label_1 = customtkinter.CTkLabel(master=self.frame_left,
                                              text="File Path",
                                              width=400,
                                              justify="center",
                                              text_font=("Roboto Medium", -12))  # font name and size in px
        self.label_1.grid(row=1, column=0, sticky="nwse")
        # self.label_1.pack(fill=tkinter.BOTH, expand=True)

        self.inside_frame_left = customtkinter.CTkFrame(master=self.frame_left)
        self.inside_frame_left.grid(row=2, column=0, sticky="nswe", padx=20, pady=20)


        self.play_button = customtkinter.CTkButton(master=self.frame_left, text='Play', command=self.play_music)
        self.play_button.grid(row=3, column=0, pady=10, padx=20)

        self.slider = customtkinter.CTkSlider(master=self.frame_left, from_=0, to=1, command=self.volume, width=210)
        self.slider.grid(row=4, column=0, pady=10, padx=20)


        self.label_mode = customtkinter.CTkButton(master=self.frame_left, text="Select File", command=self.select_file)
        self.label_mode.grid(row=6, column=0, pady=0, padx=20)





        # ============ frame_right ============

        # configure grid layout (3x7)
        self.frame_right.rowconfigure((0, 1, 2, 3), weight=1)
        self.frame_right.rowconfigure(7, weight=10)
        self.frame_right.columnconfigure((0, 1), weight=1)
        self.frame_right.columnconfigure(2, weight=0)

        self.frame_info = customtkinter.CTkFrame(master=self.frame_right)
        self.frame_info.grid(row=0, column=0, columnspan=2, rowspan=4, pady=20, padx=20, sticky="nsew")



        # ============ frame_info ============

        # configure grid layout (1x1)
        self.frame_info.rowconfigure(0, weight=1)
        self.frame_info.columnconfigure(0, weight=1)

        self.label_info_1 = customtkinter.CTkLabel(master=self.frame_info,
                                                   text="Kết Quả Dự Đoán: " ,
                                                   height=100,
                                                   text_font="Roboto 18 bold",
                                                   fg_color=("white", "gray38"),  # <- custom tuple-color
                                                   justify=tkinter.LEFT)
        self.label_info_1.grid(column=0, row=0, sticky="nwe", padx=15, pady=15)

        self.frame_result = customtkinter.CTkFrame(master=self.frame_info)
        self.frame_result.grid(row=1, column=0, columnspan=2, rowspan=4, pady=20, padx=20, sticky="nsew")

        self.frame_result.rowconfigure((0), weight=1)
        self.frame_result.columnconfigure((0,1,2), weight=1)

        self.single = customtkinter.CTkButton(master=self.frame_result, text='Đơn Tấu', command=self.predict,
                                                      height=50, text_font=("Roboto Medium", -18),fg_color="#595959", text_color="#8f8f8f")
        self.single.grid(row=0, column=0, columnspan=1, pady=20, padx=10, sticky="")

        self.two = customtkinter.CTkButton(master=self.frame_result, text='Song Tấu', command=self.predict,
                                              height=50, text_font=("Roboto Medium", -18), fg_color="#595959", text_color="#8f8f8f")
        self.two.grid(row=0, column=1, columnspan=1, pady=20, padx=10, sticky="")

        self.three = customtkinter.CTkButton(master=self.frame_result, text='Hòa Tấu', command=self.predict,
                                           height=50, text_font=("Roboto Medium", -18), fg_color="#595959", text_color="#8f8f8f")
        self.three.grid(row=0, column=2, columnspan=1, pady=20, padx=10, sticky="")

        self.progressbar = customtkinter.CTkProgressBar(master=self.frame_info)
        self.progressbar.grid(row=1, column=0, sticky="ew", padx=15, pady=15)

        # ============ frame_right ============

        self.radio_var = tkinter.IntVar(value=0)

        self.predict_button = customtkinter.CTkButton(master=self.frame_right, text='Predict', command=self.predict, height=100)
        self.predict_button.grid(row=0, column=2, columnspan=1, pady=20, padx=10, sticky="")

        # set default values

        # self.button_3.configure(state="disabled", text="Disabled CTkButton")


        self.progressbar.set(1)




    def button_event(self):
        print("Button pressed")

    def change_appearance_mode(self, new_appearance_mode):
        customtkinter.set_appearance_mode(new_appearance_mode)

    def on_closing(self, event=0):
        self.destroy()

    def play_music(self):
        if (not self.is_play):
            song_name = self.file_name
            pygame.mixer.music.load(song_name)
            pygame.mixer.music.play(loops=0)
            pygame.mixer.music.set_volume(.5)
            self.play_button.set_text("Pause")
            self.is_play = True
        else:
            pygame.mixer.music.stop();
            self.is_play = False
            self.play_button.set_text("Play")


    def volume(self,value):
        # print(value)
        pygame.mixer.music.set_volume(value)

    def select_file(self):
        filename = askopenfilename()  # show an "Open" dialog box and return the path to the selected file
        print(filename)
        sound = AudioSegment.from_mp3(filename)
        self.label_1.set_text(filename)
        self.file_name = filename
        self.plot_waveform()
        self.reset_result()

    def plot_waveform(self):

        for widget in self.inside_frame_left.winfo_children():
            widget.destroy()

        fs_rate, signal = wav.read(self.file_name)
        l_audio = len(signal.shape)
        if l_audio == 2:
            signal = signal.sum(axis=1) / 2
        N = signal.shape[0]
        secs = N / float(fs_rate)
        # print("secs", secs)
        Ts = 1.0 / fs_rate  # sampling interval in time
        # print("Timestep between samples Ts", Ts)
        t = np.arange(0, secs, Ts)  # time vector as scipy arange field / numpy.ndarray

        plt.margins(x=0)
        plt.clf()
        # create a figure
        figure = Figure(figsize=(3, 5))

        # create FigureCanvasTkAgg object
        figure_canvas = FigureCanvasTkAgg(figure, self.inside_frame_left)

        # create the toolbar
        # NavigationToolbar2Tk(figure_canvas, self.inside_frame_left)

        # create axes
        axes = figure.add_subplot()

        # create the barchart
        axes.plot(t, signal, "g")

        axes.set_facecolor("black")
        axes.get_xaxis()
        axes.get_lines()[0].set_color('yellow')
        figure_canvas.get_tk_widget().pack(side=tkinter.TOP, fill=tkinter.BOTH, expand=False)

    def predict(self):
        self.reset_result()
        model = classifier.classify(self.file_name)
        self.label_info_1.set_text("Kết Quả Dự Đoán: \n" + model[0])

        if (model[0] == "đơn tấu"):
            self.single = customtkinter.CTkButton(master=self.frame_result, text='Đơn Tấu', command=self.predict,
                                                  height=50, text_font=("Roboto Medium", -18), fg_color="orange")
            self.single.grid(row=0, column=0, columnspan=1, pady=20, padx=10, sticky="")
        elif (model[0] == "song tấu"):
            self.two = customtkinter.CTkButton(master=self.frame_result, text='Song Tấu', command=self.predict,
                                               height=50, text_font=("Roboto Medium", -18), fg_color="green")
            self.two.grid(row=0, column=1, columnspan=1, pady=20, padx=10, sticky="")
        else:
            self.three = customtkinter.CTkButton(master=self.frame_result, text='Hòa Tấu', command=self.predict,
                                                 height=50, text_font=("Roboto Medium", -18), fg_color="red")
            self.three.grid(row=0, column=2, columnspan=1, pady=20, padx=10, sticky="")


    def reset_result(self):
        self.label_info_1.set_text("Kết Quả Dự Đoán:")
        self.single = customtkinter.CTkButton(master=self.frame_result, text='Đơn Tấu', command=self.predict,
                                              height=50, text_font=("Roboto Medium", -18), fg_color="#595959",
                                              text_color="#8f8f8f")
        self.single.grid(row=0, column=0, columnspan=1, pady=20, padx=10, sticky="")

        self.two = customtkinter.CTkButton(master=self.frame_result, text='Song Tấu', command=self.predict,
                                           height=50, text_font=("Roboto Medium", -18), fg_color="#595959",
                                           text_color="#8f8f8f")
        self.two.grid(row=0, column=1, columnspan=1, pady=20, padx=10, sticky="")

        self.three = customtkinter.CTkButton(master=self.frame_result, text='Hòa Tấu', command=self.predict,
                                             height=50, text_font=("Roboto Medium", -18), fg_color="#595959",
                                             text_color="#8f8f8f")
        self.three.grid(row=0, column=2, columnspan=1, pady=20, padx=10, sticky="")

if __name__ == "__main__":
    app = App()
    app.mainloop()