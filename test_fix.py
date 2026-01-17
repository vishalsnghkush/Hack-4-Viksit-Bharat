
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))

try:
    from src.driver_monitor import DriverMonitor
    print("Import successful")
    
    monitor = DriverMonitor(camera_index=0)
    print("Initialization successful")
    
    # Try one frame
    frame, status, steer, speed, face_detected = monitor.get_driver_state()
    print(f"Frame captured. Status: {status}")
    
    monitor.release()
    print("Test passed")
except Exception as e:
    print(f"Test failed: {e}")
    import traceback
    traceback.print_exc()
