import cv2
import datetime
import os
import threading
import time

class CaptureCardRecorder:
    def __init__(self, device_index=0, output_dir="recordings"):
        """
        Initialize the capture card recorder
        
        Args:
            device_index: The index of the capture device (usually 0, 1, 2, etc.)
            output_dir: Directory to save recordings
        """
        self.device_index = device_index
        self.output_dir = output_dir
        self.recording = False
        self.cap = None
        self.out = None
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    
    def find_capture_device(self):
        """
        Find available capture devices and list them
        """
        print("Searching for capture devices...")
        index = 0
        available_devices = []
        
        while index < 10:  # Check first 10 indices
            cap = cv2.VideoCapture(index)
            if cap.read()[0]:
                available_devices.append(index)
                cap.release()
            index += 1
        
        if available_devices:
            print(f"Found capture devices at indices: {available_devices}")
        else:
            print("No capture devices found")
        
        return available_devices
    
    def initialize_capture(self):
        """
        Initialize the video capture device
        """
        # Try different backends if default doesn't work
        backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
        
        for backend in backends:
            self.cap = cv2.VideoCapture(self.device_index, backend)
            if self.cap.isOpened():
                print(f"Successfully opened capture device with backend: {backend}")
                break
        
        if not self.cap.isOpened():
            raise Exception("Failed to open capture device")
        
        # Set capture properties for EVGA XR1 Lite
        # The XR1 Lite supports up to 1080p60
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        self.cap.set(cv2.CAP_PROP_FPS, 60)
        
        # Get actual properties
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        
        print(f"Capture initialized at {self.width}x{self.height} @ {self.fps}fps")
    
    def start_recording(self, filename=None):
        """
        Start recording video
        """
        if filename is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"recording_{timestamp}.mp4"
        
        filepath = os.path.join(self.output_dir, filename)
        
        # Define codec and create VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can also try 'XVID'
        self.out = cv2.VideoWriter(filepath, fourcc, self.fps, 
                                   (self.width, self.height))
        
        if not self.out.isOpened():
            raise Exception("Failed to create video writer")
        
        self.recording = True
        print(f"Recording started: {filepath}")
        return filepath
    
    def stop_recording(self):
        """
        Stop recording video
        """
        self.recording = False
        if self.out:
            self.out.release()
            self.out = None
        print("Recording stopped")
    
    def record_with_preview(self):
        """
        Record video with preview window
        """
        print("Press 'r' to start/stop recording, 'q' to quit")
        
        try:
            self.initialize_capture()
            
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to read frame")
                    break
                
                # Write frame if recording
                if self.recording and self.out:
                    self.out.write(frame)
                
                # Add recording indicator
                if self.recording:
                    cv2.putText(frame, "RECORDING", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                # Show preview
                cv2.imshow('EVGA XR1 Lite Capture', frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    if self.recording:
                        self.stop_recording()
                    else:
                        self.start_recording()
        
        finally:
            self.cleanup()
    
    def record_without_preview(self, duration_seconds):
        """
        Record video for a specified duration without preview
        """
        try:
            self.initialize_capture()
            filepath = self.start_recording()
            
            start_time = time.time()
            frame_count = 0
            
            while (time.time() - start_time) < duration_seconds:
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to read frame")
                    break
                
                if self.out:
                    self.out.write(frame)
                    frame_count += 1
                
                # Print progress
                elapsed = time.time() - start_time
                if frame_count % self.fps == 0:  # Update every second
                    print(f"Recording... {elapsed:.1f}/{duration_seconds}s")
            
            print(f"Recorded {frame_count} frames in {elapsed:.1f} seconds")
            
        finally:
            self.stop_recording()
            self.cleanup()
    
    def cleanup(self):
        """
        Clean up resources
        """
        if self.out:
            self.out.release()
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()


def main():
    # Create recorder instance
    recorder = CaptureCardRecorder()
    
    # Find available devices
    devices = recorder.find_capture_device()
    
    if not devices:
        print("No capture devices found. Please check your connections.")
        return
    
    # If multiple devices found, let user choose
    if len(devices) > 1:
        print("\nMultiple devices found. Please select one:")
        for i, device in enumerate(devices):
            print(f"{i}: Device at index {device}")
        
        choice = int(input("Enter your choice: "))
        recorder.device_index = devices[choice]
    else:
        recorder.device_index = devices[0]
    
    # Choose recording mode
    print("\nSelect recording mode:")
    print("1. Record with preview (interactive)")
    print("2. Record for specific duration (no preview)")
    
    mode = input("Enter your choice (1 or 2): ")
    
    if mode == "1":
        recorder.record_with_preview()
    elif mode == "2":
        duration = int(input("Enter recording duration in seconds: "))
        recorder.record_without_preview(duration)
    else:
        print("Invalid choice")


if __name__ == "__main__":
    main()