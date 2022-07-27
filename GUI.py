import subprocess
import tkinter as tk
from tkinter import font
from PIL import ImageTk


def squat():
    # import Squat
    subprocess.call(" python3 Squat.py ", shell=True)

def biceps():
    # import Biceps
    subprocess.call(" python3 Biceps.py ", shell=True)

root = tk.Tk()
root.minsize(1280, 720)
frame = tk.Frame(root, background="white")
frame.pack()
my_font = font.Font(root, 'Canela 40 bold')

sqimg= ImageTk.PhotoImage(file="squat.png")
biimg= ImageTk.PhotoImage(file="biceps.png")

exbut = tk.Button(frame, height=1, width=10,
                   text="QUIT", 
                   command=quit,
                   font=my_font)
exbut.pack(side=tk.BOTTOM,expand=True, fill=tk.BOTH)
sqbut = tk.Button(frame, height=700, width=700, image=sqimg,
                   text="SQUAT",
                   command=squat,
                   font=my_font)
sqbut.pack(side=tk.LEFT,expand=True, fill=tk.BOTH)
bibut = tk.Button(frame, height=700, width=700, image=biimg,
                   text="BICEPS",
                   command=biceps,
                   font=my_font)
bibut.pack(side=tk.RIGHT,expand=True, fill=tk.BOTH)

root.mainloop()