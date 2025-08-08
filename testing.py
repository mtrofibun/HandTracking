import tkinter as tk

root = tk.Tk()
entry = tk.Entry(root)

entry.insert(0, "Default Text Here") # Inserts "Default Text Here" at index 0 (beginning)
entry.pack()
root.mainloop()