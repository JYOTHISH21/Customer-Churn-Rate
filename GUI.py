import tkinter as tk     # from tkinter import Tk for Python 3.x
from tkinter.filedialog import askopenfilename
from MODEL import model

def file_select():
    filename = askopenfilename() 
    x = model(filename)
    if x==1:
        tk.Label(master=window, text="Reulst saved as result.csv").pack()
    
window = tk.Tk()
window.title('Customer Churn')
window.geometry("900x360")
frame_a = tk.Frame()


tk.Label(master=frame_a, text="This application uses Telco data from IBM to create a model that can predict customer churn based on given features.", font=("Arial", 12)).pack()



button = tk.Button(
    master=window,
    text="Choose Data",
    width=25,
    height=3,
    bg="light grey",
    fg="black",
    command=file_select
)
button.place(relx=0.5, rely=0.75, anchor=tk.CENTER)
frame_b = tk.Frame()

frame_a.pack()
frame_b.pack()

window.mainloop()



