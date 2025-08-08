import tkinter as tk
from tkinter import ttk
def button_clicked():
    palm = entry_palm.get()
    print(palm)
root = tk.Tk()
root.title('Hand Tracking Application For Art Programs')
entry = tk.Entry(root)
mainframe = ttk.Frame(root, padding="3 3 12 12")
mainframe.grid(column=0, row=0)
root.columnconfigure(0, weight=1)
root.rowconfigure(0, weight=1)

ttk.Label(mainframe, text="Key Binds").grid(column=2, row=0)

palm_text = tk.StringVar(root, value = "Ctrl + C")
entry_palm = ttk.Entry(mainframe, width = 20, textvariable = palm_text)
entry_palm.grid(column=2, row=1)
ttk.Label(mainframe, text = "Palm Key:  ").grid(column=1, row=1)

ttk.Button(root,text = "Submit", command = button_clicked).grid(row = 2, column = 0)


for child in mainframe.winfo_children():
    child.grid_configure(padx=5, pady=5)

root.mainloop()