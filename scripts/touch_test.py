#!/usr/bin/env python3
"""Visual touch/click feedback test: draws a dot wherever a press lands,
and prints the event coordinates to stdout. Tap/click anywhere; press
Escape or the X button to quit."""

import tkinter as tk

root = tk.Tk()
root.title("Touch Test")
root.attributes("-fullscreen", True)
root.configure(bg="black")

canvas = tk.Canvas(root, bg="black", highlightthickness=0)
canvas.pack(fill="both", expand=True)

info = tk.Label(root, text="Tap or click anywhere", fg="white", bg="black",
                 font=("DejaVu Sans", 14))
info.place(x=10, y=10)

counter = [0]


def on_press(event):
    counter[0] += 1
    x, y = event.x, event.y
    r = 12
    canvas.create_oval(x - r, y - r, x + r, y + r, outline="#ffaa28", width=3)
    canvas.create_text(x, y - 20, text=f"{counter[0]}: ({x},{y})", fill="white",
                        font=("DejaVu Sans", 10))
    info.config(text=f"Last touch: ({x},{y})  |  total: {counter[0]}")
    print(f"TOUCH {counter[0]}: x={x} y={y}", flush=True)


def quit_app(event=None):
    root.destroy()


canvas.bind("<Button-1>", on_press)
root.bind("<Escape>", quit_app)

exit_btn = tk.Button(root, text="X Quit", command=quit_app, bg="#333", fg="white")
exit_btn.place(relx=0.5, rely=0.5, anchor="center", width=90, height=40)

root.mainloop()
