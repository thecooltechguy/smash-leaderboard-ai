#!/usr/bin/env python3
"""
Simple utility to test different video capture device indices.
This helps you find the correct device index for your capture card.
"""

import cv2
import sys

def test_single_device(device_index):
    """Test a single device index and display a preview window."""
    print(f"Testing device index {device_index}...")
    
    # Try different backends
    backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
    backend_names = ['DirectShow', 'Media Foundation', 'Any']
    
    for backend, name in zip(backends, backend_names):
        print(f"  Trying {name} backend...")
        
        try:
            cap = cv2.VideoCapture(device_index, backend)
            
            if not cap.isOpened():
                print(f"    ❌ Failed to open device with {name}")
                continue
            
            # Get device properties
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            print(f"    📺 Device opened: {width}x{height} @ {fps:.1f}fps")
            
            # Try to read a frame
            ret, frame = cap.read()
            if not ret or frame is None:
                print(f"    ❌ Could not read frame from device")
                cap.release()
                continue
            
            print(f"    ✅ Successfully reading frames!")
            print(f"    📊 Frame shape: {frame.shape}")
            
            # Show preview window
            window_name = f"Device {device_index} Preview - Press 'q' to quit, 's' to save screenshot"
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, 800, 600)
            
            frame_count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("    ❌ Lost connection to device")
                    break
                
                # Add device info overlay
                info_text = f"Device {device_index} ({name}) - {width}x{height} @ {fps:.1f}fps"
                cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                frame_count += 1
                cv2.putText(frame, f"Frame: {frame_count}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                
                cv2.imshow(window_name, frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    screenshot_name = f"device_{device_index}_screenshot.jpg"
                    cv2.imwrite(screenshot_name, frame)
                    print(f"    📸 Screenshot saved as {screenshot_name}")
            
            cap.release()
            cv2.destroyAllWindows()
            
            print(f"    ✅ Device {device_index} works with {name} backend!")
            return True
            
        except Exception as e:
            print(f"    ❌ Exception with {name}: {e}")
            
        finally:
            try:
                cap.release()
            except:
                pass
    
    print(f"    ❌ Device {device_index} not working")
    return False

def test_all_devices(max_devices=10):
    """Test all devices from 0 to max_devices-1."""
    print("🔍 Testing all available video capture devices...")
    print("=" * 60)
    
    working_devices = []
    
    for i in range(max_devices):
        if test_single_device(i):
            working_devices.append(i)
        print()
    
    print("=" * 60)
    print("📋 SUMMARY:")
    
    if working_devices:
        print(f"✅ Working devices found: {working_devices}")
        print(f"\n💡 To use a specific device with the main application:")
        for device in working_devices:
            print(f"   python capture_card_processor.py --device {device}")
    else:
        print("❌ No working devices found!")
        print("   - Check that your capture card is connected")
        print("   - Install proper drivers")
        print("   - Try different USB ports")

def main():
    if len(sys.argv) > 1:
        try:
            device_index = int(sys.argv[1])
            print(f"Testing specific device index: {device_index}")
            test_single_device(device_index)
        except ValueError:
            print("Error: Device index must be a number")
            print("Usage: python test_devices.py [device_index]")
            print("       python test_devices.py           # Test all devices")
            sys.exit(1)
    else:
        test_all_devices()

if __name__ == "__main__":
    main() 