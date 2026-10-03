#!/usr/bin/env python3
"""Desktop preview of sunflower_touch_ui.py at the Pi screen's 480x320 size.
Visual check only: recording/translation need the Pi's mic and models."""

import os
import sys
import tkinter as tk

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sunflower_touch_ui import SunflowerApp

root = tk.Tk()
app = SunflowerApp(root)
root.attributes("-fullscreen", False)
root.geometry("480x320+120+120")
root.resizable(False, False)
# Auto-close so a forgotten preview never lingers.
root.after(int(sys.argv[1]) * 1000 if len(sys.argv) > 1 else 60000, root.destroy)
root.mainloop()
