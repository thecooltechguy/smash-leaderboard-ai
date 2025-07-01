import re
import cv2
import numpy as np
import datetime
import os
import threading
import time
import argparse
from enum import Enum
import math
from typing import List, Optional
from google import genai
from google.genai import types
from pydantic import BaseModel
from supabase import create_client, Client
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class GameState(Enum):
    WAITING = "waiting"
    READY_DETECTED = "ready_detected"
    RECORDING = "recording"
    GAME_END_DETECTED = "game_end_detected"

class PlayerStats(BaseModel):
    smash_character: str
    player_name: str
    is_cpu: bool
    total_kos: int
    total_falls: int
    total_sds: int
    has_won: bool

# Initialize Gemini client
try:
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        raise ValueError("GEMINI_API_KEY not found in environment variables")
    
    gemini_client = genai.Client(
        api_key=gemini_api_key,
    )
    gemini_model = "gemini-2.5-pro-preview-06-05"
except Exception as e:
    print(f"Warning: Failed to initialize Gemini client: {e}")
    gemini_client = None

# Initialize Supabase client
try:
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    
    if not supabase_url or not supabase_key:
        raise ValueError("SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY not found in environment variables")
    
    supabase_client: Client = create_client(supabase_url, supabase_key)
except Exception as e:
    print(f"Warning: Failed to initialize Supabase client: {e}")
    supabase_client = None

class SmashBrosProcessor:
    def __init__(self, device_index=0, output_dir="matches", test_mode=False, test_video_path=None,
                 center_region_top=0.3, center_region_bottom=0.7, center_region_left=0.1, center_region_right=0.9,
                 game_region_top=0.1, game_region_bottom=0.5, game_region_left=0.2, game_region_right=0.8,
                 consecutive_black_threshold_secs=0.5, play_video=False):
        """
        Initialize the Smash Bros match processor
        
        Args:
            device_index: The index of the capture device
            output_dir: Directory to save match recordings
            test_mode: Whether to run in test mode with existing video
            test_video_path: Path to test video file
            center_region_top: Top boundary for center region (0.0-1.0)
            center_region_bottom: Bottom boundary for center region (0.0-1.0)
            center_region_left: Left boundary for center region (0.0-1.0)
            center_region_right: Right boundary for center region (0.0-1.0)
            game_region_top: Top boundary for game region (0.0-1.0)
            game_region_bottom: Bottom boundary for game region (0.0-1.0)
            game_region_left: Left boundary for game region (0.0-1.0)
            game_region_right: Right boundary for game region (0.0-1.0)
            consecutive_black_threshold_secs: Minimum consecutive black screen duration in seconds to detect as a black period
            play_video: Whether to play the video in real-time (test mode only)
        """
        self.device_index = device_index
        self.output_dir = output_dir
        self.test_mode = test_mode
        self.test_video_path = test_video_path
        self.play_video = play_video
        
        # Region boundaries (as fractions of frame dimensions)
        self.center_region_top = center_region_top
        self.center_region_bottom = center_region_bottom
        self.center_region_left = center_region_left
        self.center_region_right = center_region_right
        
        self.game_region_top = game_region_top
        self.game_region_bottom = game_region_bottom
        self.game_region_left = game_region_left
        self.game_region_right = game_region_right
        
        self.state = GameState.WAITING
        self.cap = None
        self.out = None
        self.current_match_frames = []
        self.frame_buffer = []
        self.buffer_size = 300  # 5 seconds at 60fps
        
        # Detection parameters
        self.black_screen_threshold = 0.1  # Average brightness threshold for black screen
        self.ready_confidence_threshold = 0.38 #0.7
        self.game_end_confidence_threshold = 0.7 # 0.78 # 0.6
        
        # Timing parameters
        self.frames_since_ready = 0
        self.frames_since_black = 0
        self.black_screen_duration_threshold_secs = consecutive_black_threshold_secs  # Use same threshold as black period detection
        self.ready_to_game_timeout = 600  # 10 seconds max from ready to game start
        
        # Consecutive black frame detection
        self.consecutive_black_frames = 0
        self.consecutive_black_threshold_secs = consecutive_black_threshold_secs
        self.in_black_period = False
        self.black_period_start_frame = None
        self.black_period_start_timestamp = None
        self.black_periods = []  # List to store all detected black periods
        
        # Match counter
        self.match_counter = 1
        
        # Test mode tracking
        self.current_frame_number = 0
        self.game_start_frame = None
        self.game_end_frame = None
        
        # Debug values for display
        self.last_ready_confidence = 0.0
        self.last_game_end_confidence = 0.0
        self.last_avg_brightness = 0.0
        
        # Result screen extraction tracking
        self.recording_frames = []  # Store frames during recording
        self.recording_game_end_scores = []  # Store game end confidence scores during recording
        self.current_recording_frame_index = 0  # Track frame index within current recording
        self.max_recording_frames = 3600  # Limit to ~1 minute at 60fps to prevent memory issues
        
        # Create output directory
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Create result screens directory
        self.result_screens_dir = os.path.join(output_dir, "result_screens")
        if not os.path.exists(self.result_screens_dir):
            os.makedirs(self.result_screens_dir)
    
    def initialize_capture(self):
        """Initialize video capture"""
        if self.test_mode and self.test_video_path:
            self.cap = cv2.VideoCapture(self.test_video_path)
            print(f"Initialized test mode with video: {self.test_video_path}")
            # self.cap.set(cv2.CAP_PROP_FPS, 60)
        else:
            # Try different backends for capture card
            backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
            
            for backend in backends:
                self.cap = cv2.VideoCapture(self.device_index, backend)
                if self.cap.isOpened():
                    print(f"Successfully opened capture device with backend: {backend}")
                    break
            
            if not self.cap or not self.cap.isOpened():
                raise Exception("Failed to open capture device")
            
            # Set capture properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
            self.cap.set(cv2.CAP_PROP_FPS, 60)
        
        # Get actual properties
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS)) if int(self.cap.get(cv2.CAP_PROP_FPS)) > 0 else 30
        print("FPS: ", self.fps)
        
        print(f"Capture initialized at {self.width}x{self.height} @ {self.fps}fps")
    
    def detect_ready_to_fight(self, frame):
        """
        Detect the Super Smash Bros logo (bright yellow/orange circular logo with cross)
        that appears right before the game starts
        """
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Focus on the center area where the logo appears
        h, w = frame.shape[:2]
        center_region = frame[int(h*self.center_region_top):int(h*self.center_region_bottom), int(w*self.center_region_left):int(w*self.center_region_right)]
        
        # Check if the overall frame is mostly black (characteristic of this screen)
        gray_full = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        avg_brightness = np.mean(gray_full) / 255.0
        
        # The screen should be mostly black with a bright logo
        if avg_brightness > 0.15:  # Too bright overall, not the logo screen
            return 0.0, False
        
        # Look for the bright yellow/orange logo in the center region
        center_hsv = cv2.cvtColor(center_region, cv2.COLOR_BGR2HSV)
        
        # Define range for yellow/orange colors (the logo color)
        lower_yellow_orange = np.array([15, 100, 150])  # More restrictive to catch bright yellows/oranges
        upper_yellow_orange = np.array([35, 255, 255])
        
        logo_mask = cv2.inRange(center_hsv, lower_yellow_orange, upper_yellow_orange)
        
        # Calculate the percentage of yellow/orange pixels in center region
        logo_ratio = np.sum(logo_mask > 0) / (logo_mask.shape[0] * logo_mask.shape[1])
        
        # Look for circular/round bright areas (the logo is circular)
        gray_center = cv2.cvtColor(center_region, cv2.COLOR_BGR2GRAY)
        
        # Find very bright areas (the glowing logo)
        bright_mask = gray_center > 180
        bright_ratio = np.sum(bright_mask) / (bright_mask.shape[0] * bright_mask.shape[1])
        
        # Look for the cross pattern within the bright area
        # The cross creates dark lines through the bright circular logo
        if bright_ratio > 0.02:  # Only check for cross if we have enough bright pixels
            # Apply morphological operations to find the cross pattern
            kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))
            cross_enhanced = cv2.morphologyEx(bright_mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
            cross_ratio = np.sum(cross_enhanced > 0) / (cross_enhanced.shape[0] * cross_enhanced.shape[1])
        else:
            cross_ratio = 0.0
        
        # Combine all metrics
        # - High logo color ratio
        # - Sufficient bright area (the glowing effect)
        # - Dark background (low overall brightness)
        # - Cross pattern detection
        background_darkness = max(0, (0.15 - avg_brightness) / 0.15)  # Higher score for darker backgrounds
        
        confidence = (logo_ratio * 3 + bright_ratio * 2 + cross_ratio * 1 + background_darkness * 1) / 7
        
        # print(f"Logo detection - Logo ratio: {logo_ratio:.4f}, Bright ratio: {bright_ratio:.4f}, Cross ratio: {cross_ratio:.4f}, Background darkness: {background_darkness:.4f}, Overall brightness: {avg_brightness:.4f}")
        # print(f"Confidence: {confidence:.4f}")
        return confidence, confidence > self.ready_confidence_threshold
    
    def detect_game_end(self, frame):
        """
        Detect game end by looking for 'GAME!' text or victory screen elements
        """
        # Convert to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Focus on upper center area where "GAME!" typically appears
        h, w = frame.shape[:2]
        game_region = frame[int(h*self.game_region_top):int(h*self.game_region_bottom), int(w*self.game_region_left):int(w*self.game_region_right)]
        
        # Look for bright yellow/white text (typical of "GAME!" text)
        gray_game = cv2.cvtColor(game_region, cv2.COLOR_BGR2GRAY)
        
        # Look for very bright areas (GAME! text is usually very bright)
        bright_mask = gray_game > 200
        bright_ratio = np.sum(bright_mask) / (bright_mask.shape[0] * bright_mask.shape[1])
        
        # Look for result screen UI elements (usually has specific color patterns)
        # game_hsv = cv2.cvtColor(game_region, cv2.COLOR_BGR2HSV)
        
        # Check for blue UI elements (common in results screen)
        # lower_blue = np.array([100, 50, 50])
        # upper_blue = np.array([130, 255, 255])
        # blue_mask = cv2.inRange(game_hsv, lower_blue, upper_blue)
        # blue_ratio = np.sum(blue_mask > 0) / (blue_mask.shape[0] * blue_mask.shape[1])
        
        # Combine metrics
        confidence = bright_ratio #(bright_ratio + blue_ratio * 0.5)
        
        return confidence, confidence >= self.game_end_confidence_threshold
    
    def is_black_screen(self, frame):
        """
        Detect if the frame is mostly black
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        avg_brightness = np.mean(gray) / 255.0
        return avg_brightness, avg_brightness < self.black_screen_threshold
    
    def format_timestamp(self, frame_number):
        """
        Convert frame number to timestamp format (HH:MM:SS.mmm)
        """
        if self.fps <= 0:
            return f"Frame {frame_number}"
        
        total_seconds = frame_number / self.fps
        hours = int(total_seconds // 3600)
        minutes = int((total_seconds % 3600) // 60)
        seconds = total_seconds % 60
        
        return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}"
    
    def timestamp_to_frame(self, timestamp_str):
        """
        Convert timestamp in mm:ss format to frame number
        """
        try:
            parts = timestamp_str.split(':')
            if len(parts) == 2:
                minutes, seconds = parts
                total_seconds = int(minutes) * 60 + float(seconds)
            elif len(parts) == 3:
                hours, minutes, seconds = parts
                total_seconds = int(hours) * 3600 + int(minutes) * 60 + float(seconds)
            else:
                raise ValueError("Invalid timestamp format")
                
            return int(total_seconds * self.fps)
        except (ValueError, IndexError) as e:
            raise ValueError(f"Invalid timestamp format '{timestamp_str}'. Use mm:ss or hh:mm:ss format") from e
    
    def test_threshold_at_timestamp(self, timestamp_str):
        """
        Test detection thresholds at a specific timestamp
        """
        if not self.test_mode or not self.test_video_path:
            print("Error: test-threshold requires test mode with video")
            return
        
        try:
            # Initialize capture first to get fps
            self.initialize_capture()
            
            # Now convert timestamp to frame number
            target_frame = self.timestamp_to_frame(timestamp_str)
            print(f"Seeking to timestamp {timestamp_str} (frame {target_frame})")
            
            # Seek to the target frame
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
            
            # Read the frame
            ret, frame = self.cap.read()
            if not ret:
                print(f"Error: Could not read frame at timestamp {timestamp_str}")
                return
            
            # Get actual frame position (might be slightly different due to keyframes)
            actual_frame = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
            actual_timestamp = self.format_timestamp(actual_frame)
            
            print(f"Actual frame: {actual_frame} ({actual_timestamp})")
            print(f"Frame dimensions: {frame.shape[1]}x{frame.shape[0]}")
            
            # Extract regions used by detection functions
            h, w = frame.shape[:2]
            
            # Center region for ready detection (same as in detect_ready_to_fight)
            center_region = frame[int(h*self.center_region_top):int(h*self.center_region_bottom), int(w*self.center_region_left):int(w*self.center_region_right)]
            
            # Game region for end detection (same as in detect_game_end)
            game_region = frame[int(h*self.game_region_top):int(h*self.game_region_bottom), int(w*self.game_region_left):int(w*self.game_region_right)]
            
            # Get detection values
            ready_confidence, ready_detected = self.detect_ready_to_fight(frame)
            game_end_confidence, game_end_detected = self.detect_game_end(frame)
            avg_brightness, is_black = self.is_black_screen(frame)
            
            # Create output filenames with timestamp
            timestamp_safe = timestamp_str.replace(':', '-')
            center_filename = f"center_region_{timestamp_safe}_frame{actual_frame}.png"
            game_filename = f"game_region_{timestamp_safe}_frame{actual_frame}.png"
            full_filename = f"full_frame_{timestamp_safe}_frame{actual_frame}.png"
            
            # Save the regions
            cv2.imwrite(center_filename, center_region)
            cv2.imwrite(game_filename, game_region)
            cv2.imwrite(full_filename, frame)
            
            # Print analysis results
            print("\n" + "="*60)
            print(f"THRESHOLD ANALYSIS AT {timestamp_str}")
            print("="*60)
            print(f"Ready Detection:")
            print(f"  Confidence: {ready_confidence:.4f}")
            print(f"  Threshold:  {self.ready_confidence_threshold:.4f}")
            print(f"  Detected:   {'✓ YES' if ready_detected else '✗ NO'}")
            print(f"  Region saved: {center_filename}")
            print()
            print(f"Game End Detection:")
            print(f"  Confidence: {game_end_confidence:.4f}")
            print(f"  Threshold:  {self.game_end_confidence_threshold:.4f}")
            print(f"  Detected:   {'✓ YES' if game_end_detected else '✗ NO'}")
            print(f"  Region saved: {game_filename}")
            print()
            print(f"Black Screen Detection:")
            print(f"  Brightness: {avg_brightness:.4f}")
            print(f"  Threshold:  {self.black_screen_threshold:.4f}")
            print(f"  Is Black:   {'✓ YES' if is_black else '✗ NO'}")
            print()
            print(f"Full Frame saved: {full_filename}")
            print("="*60)
            
            # Create annotated debug image
            debug_frame = frame.copy()
            
            # Draw region boundaries
            cv2.rectangle(debug_frame, (int(w*self.center_region_left), int(h*self.center_region_top)), (int(w*self.center_region_right), int(h*self.center_region_bottom)), (0, 255, 0), 3)  # Center region
            cv2.rectangle(debug_frame, (int(w*self.game_region_left), int(h*self.game_region_top)), (int(w*self.game_region_right), int(h*self.game_region_bottom)), (255, 0, 0), 3)  # Game region
            
            # Add labels
            cv2.putText(debug_frame, "CENTER REGION", (int(w*self.center_region_left), int(h*self.center_region_top)-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(debug_frame, "GAME REGION", (int(w*self.game_region_left), int(h*self.game_region_top)-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            
            # Add detection info
            y_pos = 50
            cv2.putText(debug_frame, f"Ready: {ready_confidence:.3f} ({'DETECTED' if ready_detected else 'NOT DETECTED'})", 
                       (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0) if ready_detected else (255, 255, 255), 2)
            y_pos += 40
            cv2.putText(debug_frame, f"GameEnd: {game_end_confidence:.3f} ({'DETECTED' if game_end_detected else 'NOT DETECTED'})", 
                       (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0) if game_end_detected else (255, 255, 255), 2)
            y_pos += 40
            cv2.putText(debug_frame, f"Brightness: {avg_brightness:.3f} ({'BLACK' if is_black else 'NOT BLACK'})", 
                       (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0) if is_black else (255, 255, 255), 2)
            
            debug_filename = f"debug_annotated_{timestamp_safe}_frame{actual_frame}.png"
            cv2.imwrite(debug_filename, debug_frame)
            print(f"Debug annotated frame saved: {debug_filename}")
            
        except ValueError as e:
            print(f"Error: {e}")
        except Exception as e:
            print(f"Unexpected error: {e}")
        finally:
            self.cleanup()
    
    def start_match_recording(self):
        """
        Start recording a new match
        """
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"match_{self.match_counter:03d}_{timestamp}.mp4"
        filepath = os.path.join(self.output_dir, filename)
        
        # Use H.264 codec for better compatibility
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.out = cv2.VideoWriter(filepath, fourcc, self.fps, (self.width, self.height))
        
        if not self.out.isOpened():
            print(f"Warning: Failed to create video writer for {filepath}")
            return None
        
        print(f"Started recording match {self.match_counter}: {filename}")
        
        # Write buffered frames (pre-game footage)
        # for buffered_frame in self.frame_buffer:
        #     self.out.write(buffered_frame)
        
        self.current_match_frames = []
        
        # Reset result screen tracking for new match
        self.recording_frames = []
        self.recording_game_end_scores = []
        self.current_recording_frame_index = 0
        
        return filepath
    
    def stop_match_recording(self):
        """
        Stop recording current match
        """
        if self.out:
            self.out.release()
            self.out = None
            print(f"Stopped recording match {self.match_counter}")
            
            # Extract result screens if we have recorded frames
            self.extract_result_screens()
            
            self.match_counter += 1
    
    def extract_result_screens(self):
        """
        Extract and save result screen frames from the recorded match
        """
        if not self.recording_frames or not self.recording_game_end_scores:
            print("No recorded frames or game end scores to process for result screens")
            return
        
        print(f"Analyzing {len(self.recording_frames)} recorded frames for result screen extraction...")
        
        # Find the last frame with highest game end confidence above threshold
        best_frame_index = -1
        best_confidence = 0.0
        
        # Search backwards through the scores to find the last frame with high confidence
        for i in range(len(self.recording_game_end_scores) - 1, -1, -1):
            confidence = self.recording_game_end_scores[i]
            if confidence >= self.game_end_confidence_threshold and confidence > best_confidence:
                best_confidence = confidence
                best_frame_index = i
                break  # We want the last (most recent) frame with high confidence
        
        if best_frame_index == -1:
            print("No frame found with game end confidence above threshold for result screens")
            return
        
        # Extract frames from the best frame to the end
        result_frames = self.recording_frames[best_frame_index:]
        
        if len(result_frames) < 30:  # Less than 0.5 seconds at 60fps
            print(f"Result screen sequence too short ({len(result_frames)} frames), skipping")
            return
        
        # Create result screen video filename
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        result_filename = f"result_screen_match_{self.match_counter:03d}_{timestamp}.mp4"
        result_filepath = os.path.join(self.result_screens_dir, result_filename)
        
        # Create video writer for result screens
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        result_out = cv2.VideoWriter(result_filepath, fourcc, self.fps, (self.width, self.height))
        
        if not result_out.isOpened():
            print(f"Warning: Failed to create result screen video writer for {result_filepath}")
            return
        
        # Write result screen frames
        for frame in result_frames:
            result_out.write(frame)
        
        result_out.release()
        
        # Calculate duration
        duration_seconds = len(result_frames) / self.fps if self.fps > 0 else 0
        
        print(f"Saved result screens: {result_filename}")
        print(f"  Duration: {duration_seconds:.2f} seconds ({len(result_frames)} frames)")
        print(f"  Starting from frame with confidence: {best_confidence:.3f}")
        print(f"  Frame index in match: {best_frame_index}/{len(self.recording_frames)-1}")
        
        # Extract player stats and save to database (only when NOT in test mode)
        if not self.test_mode:
            # Run match stats processing in background thread to avoid blocking frame processing
            def process_match_results_background():
                try:
                    print("\nProcessing match results in background...")
                    
                    # Extract player stats using Gemini API
                    match_stats = self.get_match_stats(result_filepath)
                    
                    if match_stats:
                        # Save match stats to database
                        self.save_match_stats(match_stats)
                    else:
                        print("Failed to extract match stats, skipping database save")
                except Exception as e:
                    print(f"Error in background match processing: {e}")
            
            # Start background thread
            background_thread = threading.Thread(target=process_match_results_background, daemon=True)
            background_thread.start()
            print("Match results processing started in background thread")
    
    def process_frame(self, frame):
        """
        Process a single frame and update state machine
        """
        self.current_frame_number += 1
        
        # Get detection values for debugging
        ready_confidence, ready_detected = self.detect_ready_to_fight(frame)
        game_end_confidence, game_end_detected = self.detect_game_end(frame)
        avg_brightness, is_black = self.is_black_screen(frame)
        
        # Store for display
        self.last_ready_confidence = ready_confidence
        self.last_game_end_confidence = game_end_confidence
        self.last_avg_brightness = avg_brightness
        
        # Print debug info in test mode (every 30 frames to avoid spam)
        if self.test_mode and self.current_frame_number % 30 == 0:
            timestamp = self.format_timestamp(self.current_frame_number)
            print(f"[DEBUG {timestamp}] Ready: {ready_confidence:.3f}, GameEnd: {game_end_confidence:.3f}, Brightness: {avg_brightness:.3f}, State: {self.state.value}")
        
        # Consecutive black frame detection
        if is_black:
            self.consecutive_black_frames += 1
            
            # Start of a new black period
            if not self.in_black_period and self.consecutive_black_frames >= (self.consecutive_black_threshold_secs * self.fps):
                self.in_black_period = True
                self.black_period_start_frame = self.current_frame_number - int(self.consecutive_black_threshold_secs * self.fps) + 1
                self.black_period_start_timestamp = self.format_timestamp(self.black_period_start_frame)
                
                if self.test_mode:
                    print(f"[BLACK PERIOD START] Frame {self.black_period_start_frame} ({self.black_period_start_timestamp}) - Brightness: {avg_brightness:.3f}")
        else:
            # End of black period
            if self.in_black_period:
                black_period_end_frame = self.current_frame_number - 1
                black_period_end_timestamp = self.format_timestamp(black_period_end_frame)
                
                # Calculate duration
                duration_frames = black_period_end_frame - self.black_period_start_frame + 1
                duration_seconds = duration_frames / self.fps if self.fps > 0 else 0
                
                # Store the black period
                black_period = {
                    'start_frame': self.black_period_start_frame,
                    'end_frame': black_period_end_frame,
                    'start_timestamp': self.black_period_start_timestamp,
                    'end_timestamp': black_period_end_timestamp,
                    'duration_frames': duration_frames,
                    'duration_seconds': duration_seconds
                }
                self.black_periods.append(black_period)
                
                if self.test_mode:
                    print(f"[BLACK PERIOD END] Frame {black_period_end_frame} ({black_period_end_timestamp}) - Duration: {duration_seconds:.2f}s ({duration_frames} frames)")
                
                # Reset black period tracking
                self.in_black_period = False
                self.black_period_start_frame = None
                self.black_period_start_timestamp = None
            
            self.consecutive_black_frames = 0
        
        # Add frame to buffer (for pre-game footage)
        self.frame_buffer.append(frame.copy())
        if len(self.frame_buffer) > self.buffer_size:
            self.frame_buffer.pop(0)
        
        # State machine logic
        if self.state == GameState.WAITING:
            if ready_detected:
                print("Detected 'READY TO FIGHT!' - Starting recording immediately...")
                if self.test_mode:
                    timestamp = self.format_timestamp(self.current_frame_number)
                    print(f"  [TEST MODE] Ready detected at: {timestamp} (Frame {self.current_frame_number}) - Confidence: {ready_confidence:.3f}")
                
                # Start recording immediately
                self.start_match_recording()
                self.game_start_frame = self.current_frame_number
                if self.test_mode:
                    timestamp = self.format_timestamp(self.current_frame_number)
                    print(f"  [TEST MODE] *** GAME START at: {timestamp} (Frame {self.current_frame_number}) ***")
                self.state = GameState.RECORDING
                self.frames_since_black = 0
        
        elif self.state == GameState.READY_DETECTED:
            # This state is no longer used - we go directly to RECORDING
            pass
        
        elif self.state == GameState.RECORDING:
            # Write frame to video
            if self.out:
                self.out.write(frame)
            
            # Store frame and game end confidence for result screen extraction
            self.recording_frames.append(frame.copy())
            self.recording_game_end_scores.append(game_end_confidence)
            self.current_recording_frame_index += 1
            
            # Limit memory usage by keeping only the most recent frames
            if len(self.recording_frames) > self.max_recording_frames:
                self.recording_frames.pop(0)
                self.recording_game_end_scores.pop(0)
            
            # Check for game end using sustained black screen (3 seconds)
            if is_black:
                self.frames_since_black += 1
                if self.frames_since_black > (self.black_screen_duration_threshold_secs * self.fps):
                    print("Detected sustained black screen - ending recording...")
                    self.stop_match_recording()
                    self.game_end_frame = self.current_frame_number
                    if self.test_mode:
                        timestamp = self.format_timestamp(self.current_frame_number)
                        print(f"  [TEST MODE] *** GAME END at: {timestamp} (Frame {self.current_frame_number}) *** - Brightness: {avg_brightness:.3f}")
                        if self.game_start_frame:
                            duration_frames = self.game_end_frame - self.game_start_frame
                            duration_seconds = duration_frames / self.fps if self.fps > 0 else 0
                            print(f"  [TEST MODE] Match duration: {duration_seconds:.2f} seconds ({duration_frames} frames)")
                    self.state = GameState.WAITING
                    self.frames_since_black = 0
                    print("Waiting for next match...")
            else:
                self.frames_since_black = 0
        
        elif self.state == GameState.GAME_END_DETECTED:
            # This state is no longer used - game end is detected directly in RECORDING state
            pass
    
    def run(self):
        """
        Main processing loop
        """
        print("Starting Smash Bros match processor...")
        print(f"State: {self.state.value}")
        
        if self.test_mode and not self.play_video:
            print("Test mode: Fast offline processing (no video display)")
        elif self.test_mode and self.play_video:
            print("Test mode: Real-time video playback")
        
        try:
            self.initialize_capture()
            
            frame_count = 0
            start_time = time.time()
            
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    if self.test_mode:
                        print("Reached end of test video")
                        break
                    else:
                        print("Failed to read frame")
                        continue
                
                # Process the frame
                self.process_frame(frame)
                frame_count += 1
                
                # Handle display and timing based on mode
                if self.test_mode and not self.play_video:
                    # Fast offline processing - no display, no delays
                    # Print progress every 1000 frames to show activity
                    if frame_count % 1000 == 0:
                        elapsed = time.time() - start_time
                        fps_processed = frame_count / elapsed if elapsed > 0 else 0
                        print(f"Processed {frame_count} frames ({fps_processed:.1f} fps) - State: {self.state.value}")
                else:
                    # Real-time playback or live capture - show display
                    # Create display frame with status
                    display_frame = frame.copy()
                    
                    # Add status overlay
                    status_text = f"State: {self.state.value}"
                    cv2.putText(display_frame, status_text, (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    
                    if self.state == GameState.RECORDING:
                        cv2.putText(display_frame, f"RECORDING MATCH {self.match_counter}", (10, 70), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    
                    # Add debug info overlay in test mode
                    if self.test_mode:
                        y_offset = 110 if self.state == GameState.RECORDING else 70
                        
                        # Ready confidence
                        ready_color = (0, 255, 0) if self.last_ready_confidence > self.ready_confidence_threshold else (255, 255, 255)
                        cv2.putText(display_frame, f"Ready: {self.last_ready_confidence:.3f}", (10, y_offset), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, ready_color, 2)
                        
                        # Game end confidence
                        game_end_color = (0, 255, 0) if self.last_game_end_confidence > self.game_end_confidence_threshold else (255, 255, 255)
                        cv2.putText(display_frame, f"GameEnd: {self.last_game_end_confidence:.3f}", (10, y_offset + 35), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, game_end_color, 2)
                        
                        # Brightness
                        brightness_color = (0, 255, 0) if self.last_avg_brightness < self.black_screen_threshold else (255, 255, 255)
                        cv2.putText(display_frame, f"Brightness: {self.last_avg_brightness:.3f}", (10, y_offset + 70), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, brightness_color, 2)
                        
                        # Consecutive black frames
                        black_period_color = (255, 0, 0) if self.in_black_period else (255, 255, 255)
                        cv2.putText(display_frame, f"Black frames: {self.consecutive_black_frames} {'[IN BLACK PERIOD]' if self.in_black_period else ''}", 
                                   (10, y_offset + 105), cv2.FONT_HERSHEY_SIMPLEX, 0.7, black_period_color, 2)
                        
                        # Frame number and timestamp
                        timestamp = self.format_timestamp(self.current_frame_number)
                        cv2.putText(display_frame, f"Frame: {self.current_frame_number} ({timestamp})", (10, y_offset + 140), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
                    
                    # Show preview (smaller window)
                    small_frame = cv2.resize(display_frame, (960, 540))
                    cv2.imshow('Smash Bros Match Processor', small_frame)
                    
                    # Handle key presses and timing
                    if self.test_mode and self.play_video:
                        # Real-time playback - wait for proper frame timing
                        wait_time = max(1, int(1000 / self.fps))
                        key = cv2.waitKey(wait_time) & 0xFF
                    else:
                        # Live capture - minimal delay
                        key = cv2.waitKey(1) & 0xFF
                    
                    if key == ord('q'):
                        break
                    elif key == ord('r'):  # Manual reset
                        print("Manual state reset")
                        self.state = GameState.WAITING
                        if self.out:
                            self.stop_match_recording()
                    
                    # Print progress every 5 seconds for live capture or real-time playback
                    if frame_count % (self.fps * 5) == 0:
                        elapsed = time.time() - start_time
                        print(f"Processed {frame_count} frames in {elapsed:.1f}s - State: {self.state.value}")
        
        finally:
            self.cleanup()
    
    def print_black_periods_summary(self):
        """
        Print a summary of all detected black periods
        """
        if not self.black_periods:
            print("\n[BLACK PERIODS SUMMARY] No black periods detected")
            return
        
        print(f"\n[BLACK PERIODS SUMMARY] Detected {len(self.black_periods)} black periods:")
        print("="*80)
        
        total_black_duration = 0
        for i, period in enumerate(self.black_periods, 1):
            print(f"Period {i:2d}: {period['start_timestamp']} - {period['end_timestamp']} "
                  f"(Duration: {period['duration_seconds']:6.2f}s, Frames: {period['duration_frames']:4d})")
            total_black_duration += period['duration_seconds']
        
        print("="*80)
        print(f"Total black screen time: {total_black_duration:.2f} seconds")
        
        # Calculate video statistics if we have frame info
        if self.current_frame_number > 0 and self.fps > 0:
            total_video_duration = self.current_frame_number / self.fps
            black_percentage = (total_black_duration / total_video_duration) * 100
            print(f"Total video duration: {total_video_duration:.2f} seconds")
            print(f"Black screen percentage: {black_percentage:.1f}%")
        
        print("="*80)
    
    def update_elo(self, rating_a: float,
                   rating_b: float,
                   winner: str,
                   k: int = 32) -> tuple[int, int]:
        """
        Return the new (rating_a, rating_b) after one game.

        Parameters
        ----------
        rating_a : current Elo for Player A
        rating_b : current Elo for Player B
        winner   : 'A', 'B', or 'draw'
        k        : K-factor (default 32)

        Returns
        -------
        tuple[int, int] : New ratings as integers for (Player A, Player B)

        >>> update_elo(1400, 1000, 'A')
        (1403, 997)
        """
        # Expected scores (logistic curve)
        expected_a = 1 / (1 + 10 ** ((rating_b - rating_a) / 400))
        expected_b = 1 - expected_a

        # Actual scores
        if winner.upper() == 'A':
            score_a, score_b = 1.0, 0.0
        elif winner.upper() == 'B':
            score_a, score_b = 0.0, 1.0
        elif winner.lower() in ('draw', 'tie', 'd'):
            score_a = score_b = 0.5
        else:
            raise ValueError("winner must be 'A', 'B', or 'draw'")

        # Rating updates
        new_rating_a = rating_a + k * (score_a - expected_a)
        new_rating_b = rating_b + k * (score_b - expected_b)

        # Return as integers
        return round(new_rating_a), round(new_rating_b)
    
    def get_match_stats(self, match_results_video_filepath: str, slowdown_factor: int = 5) -> Optional[List[PlayerStats]]:
        """
        Extract player stats from a match results video using Gemini API
        """
        if not gemini_client:
            print("Warning: Gemini client not available, skipping stats extraction")
            return None
        
        try:
            print(f"Extracting player stats from result screen video: {match_results_video_filepath}")
            
            # First use ffmpeg to slow down the video
            final_video_filepath = "./current_match_results_video.mp4"
            ffmpeg_cmd = f"ffmpeg -y -an -i \"{match_results_video_filepath}\" -vf \"setpts={slowdown_factor}*PTS\" \"{final_video_filepath}\" -loglevel quiet"
            
            if os.system(ffmpeg_cmd) != 0:
                print("Error: Failed to process video with ffmpeg")
                return None
            
            # Upload file to Gemini
            print("Uploading video to Gemini API...")
            file = gemini_client.files.upload(file=final_video_filepath)
            
            # Wait for file to be processed
            while True:
                file_info = gemini_client.files.get(name=file.name)
                if file_info.state == "ACTIVE":
                    break
                time.sleep(1)
            
            # Prepare content for Gemini
            contents = [    
                file,
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_text(text="""Here is a video recording of the results screen of a super smash bros ultimate match.

Output the following information about the game's results as valid json following this schema (where it's a list of json objects -- one for each player in the match):

```
[
{
\"smash_character\" : string,
\"player_name\" : string,
\"is_cpu\" : boolean,
\"total_kos\" : int,
\"total_falls\" : int,
\"total_sds\" : int,
\"has_won\" : boolean
},
...
]
```

keep the following in mind:

- the total number of KOs is an integer number located to the right of the label, and cannot be null. if you can't see a number next to the \"KOs\" label, then you can count the number of mini character icons shown under the \"KOs\" section of the character card
- total number of falls is an integer number located to the right of the label, and cannot be null. if you can't see a number next to the \"Falls\" label, then you can count the number of mini character icons shown under the \"Falls\" section of the character card
- total number of SDs is an integer number located to the right of the label, and cannot be null
- \"has_won\" denotes whether or not the character won (labeled with a gold-colored number 1 at the top right of the player card. if there is no such number ranking on the top right, then the character did not win; for \"no contest\" matches, no character wins)"""),
                    ],
                ),
            ]

            print("Analyzing video with Gemini API...")
            response = gemini_client.models.generate_content(
                model=gemini_model,
                config=types.GenerateContentConfig(
                    response_mime_type='application/json',
                    response_schema=list[PlayerStats],
                ),
                contents=contents,
            )
            
            # Clean up uploaded file
            gemini_client.files.delete(name=file.name)
            
            # Clean up temporary video file
            try:
                os.remove(final_video_filepath)
            except:
                pass
            
            print(f"Successfully extracted stats for {len(response.parsed)} players")
            return response.parsed
            
        except Exception as e:
            print(f"Error extracting match stats: {e}")
            return None
    
    def get_player(self, player_name: str) -> Optional[dict]:
        """Get or create a player in the database"""
        if not supabase_client:
            return None
        
        try:
            response = (
                supabase_client.table("players")
                .upsert({"name": player_name}, on_conflict="name")
                .execute()
            )
            return response.data[0]
        except Exception as e:
            print(f"Error getting/creating player {player_name}: {e}")
            return None
        
    def update_player_elo(self, player_id: str, elo: int):
        """Update a player's ELO in the database"""
        if not supabase_client:
            return
        
        supabase_client.table("players").update({"elo": elo}).eq("id", player_id).execute()
    
    def create_match(self) -> Optional[int]:
        """Create a new match in the database"""
        if not supabase_client:
            return None
        
        try:
            response = (
                supabase_client.table("matches")
                .insert({})
                .execute()
            )
            return response.data[0]['id']
        except Exception as e:
            print(f"Error creating match: {e}")
            return None
        
    def save_match_stats(self, stats: List[PlayerStats], match_id: Optional[int] = None):
        """Save match stats to the database"""
        print(stats)
        if not supabase_client:
            print("Warning: Supabase client not available, skipping database save")
            return
        
        match_is_no_contest = all(not stat.has_won for stat in stats)
        if match_is_no_contest:
            print("Match is a no contest, skipping database save")
            return
        
        match_has_cpu = any(stat.is_cpu for stat in stats)
        if match_has_cpu:
            print("Match has CPU, skipping database save")
            return
        
        match_has_unknown_players = False
        for stat in stats:
            # check if player name matches the following pattern: "Player <Number>" or "P<Number>" or "P <Number>"
            if re.match(r"^Player \d+$", stat.player_name) or re.match(r"^P\d+$", stat.player_name) or re.match(r"^P \d+$", stat.player_name):
                match_has_unknown_players = True
                break
        
        if match_has_unknown_players:
            print("Match has unknown players (Player 1,2,3,etc.), skipping database save")
            return

        try:
            if match_id is None:
                match_id = self.create_match()
                if match_id is None:
                    return
            
            players = []
            winners = []
            
            print(f"Saving match stats to database (Match ID: {match_id})")
            
            for stat in stats:
                player = self.get_player(stat.player_name)
                if player is None:
                    continue
                
                # Save match participant
                response = (
                    supabase_client.table("match_participants")
                    .insert({
                        "player": player['id'], 
                        "smash_character": stat.smash_character.upper(),
                        "is_cpu": stat.is_cpu,
                        "total_kos": stat.total_kos,
                        "total_falls": stat.total_falls,
                        "total_sds": stat.total_sds,
                        "has_won": stat.has_won,
                        "match_id": match_id,
                    })
                    .execute()
                )
                
                players.append({
                    "id": player['id'], 
                    "elo": player['elo'], 
                    "name": player['name'], 
                    "character": stat.smash_character,
                    "has_won": stat.has_won,
                    "kos": stat.total_kos,
                    "falls": stat.total_falls,
                    "sds": stat.total_sds
                })
                
                if stat.has_won:
                    winners.append(player['name'])
            
            # Print match results
            print("\n" + "="*60)
            print("MATCH RESULTS")
            print("="*60)
            
            if winners:
                print(f"🏆 Winner(s): {', '.join(winners)}")
            else:
                print("🤝 No Contest")
            
            print("\nPlayer Stats:")
            for player in players:
                status = "🏆 WINNER" if player['has_won'] else ""
                print(f"  {player['name']} ({player['character']}) - KOs: {player['kos']}, Falls: {player['falls']}, SDs: {player['sds']} {status}")
            
            # Update ELO ratings for 1v1 matches
            if len(stats) == 2:
                print("\n1v1 Match detected - Updating ELO ratings:")
                
                old_elo_1 = players[0]['elo']
                old_elo_2 = players[1]['elo']
                
                winner_index = 1 if players[0]['has_won'] else 2
                new_elo_1, new_elo_2 = self.update_elo(old_elo_1, old_elo_2, 'A' if winner_index == 1 else 'B')
                
                self.update_player_elo(players[0]['id'], new_elo_1)
                self.update_player_elo(players[1]['id'], new_elo_2)
                
                # Print ELO changes
                elo_change_1 = new_elo_1 - old_elo_1
                elo_change_2 = new_elo_2 - old_elo_2
                
                print(f"  {players[0]['name']}: {old_elo_1} → {new_elo_1} ({elo_change_1:+d})")
                print(f"  {players[1]['name']}: {old_elo_2} → {new_elo_2} ({elo_change_2:+d})")
            
            print("="*60)
            
        except Exception as e:
            print(f"Error saving match stats: {e}")
    
    def cleanup(self):
        """
        Clean up resources
        """
        # Handle any ongoing black period at the end
        if self.in_black_period:
            black_period_end_frame = self.current_frame_number
            black_period_end_timestamp = self.format_timestamp(black_period_end_frame)
            
            duration_frames = black_period_end_frame - self.black_period_start_frame + 1
            duration_seconds = duration_frames / self.fps if self.fps > 0 else 0
            
            black_period = {
                'start_frame': self.black_period_start_frame,
                'end_frame': black_period_end_frame,
                'start_timestamp': self.black_period_start_timestamp,
                'end_timestamp': black_period_end_timestamp,
                'duration_frames': duration_frames,
                'duration_seconds': duration_seconds
            }
            self.black_periods.append(black_period)
            
            if self.test_mode:
                print(f"[BLACK PERIOD END] Frame {black_period_end_frame} ({black_period_end_timestamp}) - Duration: {duration_seconds:.2f}s ({duration_frames} frames) [END OF VIDEO]")
        
        # Print summary of all black periods
        self.print_black_periods_summary()
        
        if self.out:
            self.out.release()
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description='Super Smash Bros Match Processor')
    parser.add_argument('--test', action='store_true', help='Run in test mode with existing video')
    parser.add_argument('--video', type=str, help='Path to test video file')
    parser.add_argument('--play-video', action='store_true', help='Play video in real-time during test mode (default: fast offline processing)')
    parser.add_argument('--test-threshold', type=str, help='Test detection thresholds at specific timestamp (mm:ss or hh:mm:ss format)')
    parser.add_argument('--device', type=int, default=0, help='Capture device index (default: 0)')
    parser.add_argument('--output', type=str, default='matches', help='Output directory (default: matches)')
    
    # Region boundary arguments
    parser.add_argument('--center-region-top', type=float, default=0.3, help='Top boundary for center region (0.0-1.0, default: 0.3)')
    parser.add_argument('--center-region-bottom', type=float, default=0.7, help='Bottom boundary for center region (0.0-1.0, default: 0.7)')
    parser.add_argument('--center-region-left', type=float, default=0.4, help='Left boundary for center region (0.0-1.0, default: 0.1)')
    parser.add_argument('--center-region-right', type=float, default=0.6, help='Right boundary for center region (0.0-1.0, default: 0.9)')
    
    parser.add_argument('--game-region-top', type=float, default=0.27, help='Top boundary for game region (0.0-1.0, default: 0.1)')
    parser.add_argument('--game-region-bottom', type=float, default=0.54, help='Bottom boundary for game region (0.0-1.0, default: 0.5)')
    parser.add_argument('--game-region-left', type=float, default=0.2, help='Left boundary for game region (0.0-1.0, default: 0.2)')
    parser.add_argument('--game-region-right', type=float, default=0.8, help='Right boundary for game region (0.0-1.0, default: 0.8)')
    
    # Black frame detection arguments
    parser.add_argument('--black-frame-threshold-secs', type=float, default=0.5, help='Minimum consecutive black screen duration in seconds to detect as a black period (default: 0.5)')
    
    args = parser.parse_args()
    
    if args.test and not args.video:
        print("Error: Test mode requires --video parameter")
        return
    
    if args.test_threshold and not args.video:
        print("Error: --test-threshold requires --video parameter")
        return
    
    # Validate region boundaries
    def validate_region(name, top, bottom, left, right):
        if not (0.0 <= top < bottom <= 1.0):
            print(f"Error: {name} top ({top}) must be < bottom ({bottom}) and both in range 0.0-1.0")
            return False
        if not (0.0 <= left < right <= 1.0):
            print(f"Error: {name} left ({left}) must be < right ({right}) and both in range 0.0-1.0")
            return False
        return True
    
    if not validate_region("Center region", args.center_region_top, args.center_region_bottom, 
                          args.center_region_left, args.center_region_right):
        return
    
    if not validate_region("Game region", args.game_region_top, args.game_region_bottom,
                          args.game_region_left, args.game_region_right):
        return
    
    # Create processor
    processor = SmashBrosProcessor(
        device_index=args.device,
        output_dir=args.output,
        test_mode=args.test or bool(args.test_threshold),
        test_video_path=args.video,
        center_region_top=args.center_region_top,
        center_region_bottom=args.center_region_bottom,
        center_region_left=args.center_region_left,
        center_region_right=args.center_region_right,
        game_region_top=args.game_region_top,
        game_region_bottom=args.game_region_bottom,
        game_region_left=args.game_region_left,
        game_region_right=args.game_region_right,
        consecutive_black_threshold_secs=args.black_frame_threshold_secs,
        play_video=args.play_video
    )
    
    # Handle test-threshold mode
    if args.test_threshold:
        processor.test_threshold_at_timestamp(args.test_threshold)
        return
    
    # Run processor
    try:
        processor.run()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main() 