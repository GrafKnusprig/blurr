#!/usr/bin/env python3
import tkinter as tk
from tkinter import filedialog

def main():
    # Create a hidden root window
    root = tk.Tk()
    root.withdraw()
    
    # Print out the Tkinter and Tcl/Tk versions.
    print("Tkinter version:", tk.TkVersion)
    print("Tcl/Tk version:", tk.TclVersion)
    patchlevel = root.tk.call('info', 'patchlevel')
    print("Tcl/Tk patch level:", patchlevel)
    
    # Open a file dialog to test if it's working.
    print("Opening file dialog...")
    file_path = filedialog.askopenfilename(title="Select a file for testing")
    if file_path:
        print("Selected file:", file_path)
    else:
        print("No file selected.")
    
    root.destroy()

if __name__ == "__main__":
    main()