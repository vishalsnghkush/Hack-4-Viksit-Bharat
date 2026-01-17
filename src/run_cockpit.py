import sys
from pathlib import Path
import os
import tkinter as tk
from tkinter import filedialog

# Add src to path
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

from demo_dual_stream import DualStreamNavigationSystem

def select_video_file():
    root = tk.Tk()
    root.withdraw() # Hide the main window
    root.attributes('-topmost', True) # Force to front
    print("\nOpening File Selector... (Check your taskbar if not visible)")
    file_path = filedialog.askopenfilename(
        title="Select Driving Video",
        filetypes=[("Video Files", "*.mp4 *.avi *.mov *.mkv"), ("All Files", "*.*")]
    )
    root.destroy()
    return file_path

def main():
    print("="*60)
    print("      VISION COCKPIT ACTIVATED      ")
    print("============================================================")
    print("Select Driving Scenario:")
    
    # Base source file (The reliable one)
    base_data = current_dir.parent / "data"
    default_video = base_data / "8359-208052066_small.mp4"
    sunny_video = base_data / "istockphoto-2159760544-640_adpp_is.mp4"
    
    # Define Scenarios using Filters
    scenarios = [
        {
            "name": "Rainy City (Original)",
            "file": default_video,
            "condition": "RAINY"
        },
        {
            "name": "Sunny Highway (New Video)",
            "file": sunny_video,
            "condition": "SUNNY (Original)"
        },
        {
            "name": "City Night Drive (Filter)",
            "file": default_video,
            "condition": "NIGHT"
        },
        {
            "name": "Sunny City Simulation (Filter)",
            "file": default_video,
            "condition": "SUNNY"
        },
         {
            "name": "Return Trip (Mirror Mode)",
            "file": default_video,
            "condition": "Mirror Mode"
        }
    ]
    
    for i, s in enumerate(scenarios):
        print(f"[{i+1}] {s['name']}")
        
    print(f"[{len(scenarios)+1}] BROWSE FOR VIDEO FILE... (Pop-up)")
    print("="*60)
    
    try:
        choice = input("Enter choice (default=1): ").strip()
        if not choice: choice = "1"
        idx = int(choice) - 1
        
        selected_file = ""
        condition = "CLEAR"
        
        if 0 <= idx < len(scenarios):
            selected_file = str(scenarios[idx]['file'])
            condition = scenarios[idx]['condition']
            print(f"\nLoading Scenario: {scenarios[idx]['name']}")
        elif idx == len(scenarios):
            # OPEN POPUP
            selected_file = select_video_file()
            if not selected_file:
                print("No file selected. Exiting.")
                return
                
            condition = "CUSTOM"
            # Simple heuristic
            f_lower = selected_file.lower()
            if "rain" in f_lower: condition = "RAINY"
            elif "snow" in f_lower: condition = "SNOWY"
            elif "night" in f_lower: condition = "NIGHT"
        else:
            print("Invalid choice, using default.")
            selected_file = str(scenarios[0]['file'])
            condition = scenarios[0]['condition']
            
    except Exception as e:
        print(f"Input error: {e}. Using default.")
        selected_file = str(scenarios[0]['file'])
        condition = scenarios[0]['condition']

    # Launch System
    if not os.path.exists(selected_file):
        print(f"\n[WARNING] File not found: {selected_file}")
        # Try to find the file name in data dir just in case
        alternative = base_data / Path(selected_file).name
        if alternative.exists():
             print(f"Found in data folder: {alternative}")
             selected_file = str(alternative)
        else:
             print("Using rainy default instead.\n")
             selected_file = str(default_video)

    # Use default webcam (0)
    system = DualStreamNavigationSystem(selected_file, driver_camera_index=0, weather_condition=condition)
    system.run()

if __name__ == "__main__":
    main()
