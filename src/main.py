#!/usr/bin/env python3

"""
PyTrackerVis - A CLI tool to visualize 4-channel .mod files.
Generates an oscilloscope-style .mp4 video using OpenCV and Numpy.

Prerequisites:
    1. Install ffmpeg (must be available in your system PATH)
    2. pip install numpy opencv-python scipy
"""

import argparse
import math
import os
import subprocess
import sys
import shutil
import tempfile
from pathlib import Path
import numpy as np
import cv2
from scipy.io import wavfile
from scipy import signal

# --- CONFIGURATION & COLORS ---
FPS = 60
RESOLUTION = (1920, 1080)
INFO_BAR_HEIGHT = 80
QUAD_W = RESOLUTION[0] // 2
QUAD_H = (RESOLUTION[1] - INFO_BAR_HEIGHT) // 2

# Vibrant Neon Colors (OpenCV uses BGR format)
COLORS = {
    "neon_cyan": (255, 255, 0),    # Ch1 (Top Left)
    "neon_magenta": (255, 0, 255), # Ch2 (Top Right)
    "neon_green": (0, 255, 0),     # Ch4 (Bottom Left)
    "neon_orange": (0, 165, 255)   # Ch3 (Bottom Right)
}

BG_COLOR = (20, 15, 12)       # Very dark tint
GRID_COLOR = (60, 60, 60)
BAR_COLOR = (30, 30, 35)
TEXT_COLOR = (255, 255, 255)


def check_dependencies():
    if not shutil.which("ffmpeg"):
        print("Error: 'ffmpeg' was not found in your PATH. Please install it to process audio files.")
        sys.exit(1)


def extract_audio(input_file: str, temp_wav: str):
    print(f"[*] Rendering module '{input_file}' to stereo WAV using ffmpeg...")
    cmd = [
        "ffmpeg", "-y", "-i", input_file,
        "-ar", "44100", "-ac", "2", "-c:a", "pcm_s16le", temp_wav
    ]
    result = subprocess.run(cmd, stdout=subprocess.DEVNULL)
    if result.returncode != 0:
        print("Error: ffmpeg failed to process the input file. Make sure it is a valid audio/mod file.")
        sys.exit(1)


def split_pseudo_channels(wav_path: str):
    """
    Simulates the 4 Amiga tracker channels by analyzing the stereo panning
    and splitting frequencies (Lows vs Highs) for the Left and Right tracks.
    Amiga format: Ch1(L), Ch2(R), Ch3(R), Ch4(L).
    """
    print("[*] Processing Amiga stereo panning and splitting frequencies...")
    sample_rate, data = wavfile.read(wav_path)
    
    # Normalize to -1.0 to 1.0
    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0
    elif data.ndim > 1:
        data = data.astype(np.float32) / (np.max(np.abs(data)) + 1e-6)

    # Ensure stereo
    if data.ndim == 1:
        left = data
        right = data
    else:
        left = data[:, 0]
        right = data[:, 1]

    # Create a crossover filter at 800Hz
    nyq = 0.5 * sample_rate
    low_cutoff = 800 / nyq
    sos_low = signal.butter(4, low_cutoff, btype='low', output='sos')
    sos_high = signal.butter(4, low_cutoff, btype='high', output='sos')

    ch1 = signal.sosfilt(sos_low, left)   # Left Lows
    ch2 = signal.sosfilt(sos_low, right)  # Right Lows
    ch3 = signal.sosfilt(sos_high, right) # Right Highs
    ch4 = signal.sosfilt(sos_high, left)  # Left Highs

    return sample_rate, [ch1, ch2, ch3, ch4]


def load_stems(stem_paths):
    print("[*] Loading provided stems...")
    channels = []
    sr = None
    for path in stem_paths:
        sample_rate, data = wavfile.read(path)
        if sr is None:
            sr = sample_rate
        elif sr != sample_rate:
            print(f"Error: Sample rate mismatch. Expected {sr} Hz but got {sample_rate} Hz in '{path}'.")
            sys.exit(1)
        if data.ndim > 1:
            data = data.mean(axis=1) # Convert to mono if needed
        data = data.astype(np.float32) / (np.max(np.abs(data)) + 1e-6)
        if channels and len(data) != len(channels[0]):
            print(f"Error: Channel length mismatch. Expected {len(channels[0])} samples but got {len(data)} in '{path}'.")
            sys.exit(1)
        channels.append(data)
    return sr, channels


def draw_waveform(img, x_offset, y_offset, audio_slice, color, title):
    # Draw Text
    cv2.putText(img, title, (x_offset + 20, y_offset + 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, TEXT_COLOR, 1, cv2.LINE_AA)
    
    if len(audio_slice) == 0:
        return

    # Calculate points for polyline
    center_y = y_offset + (QUAD_H // 2)
    x_coords = np.linspace(x_offset, x_offset + QUAD_W, len(audio_slice))
    y_coords = center_y - (audio_slice * (QUAD_H * 0.4)) # Scale to 80% of quadrant height

    pts = np.column_stack((x_coords, y_coords)).astype(np.int32)

    # Draw glow (thick, low alpha simulation)
    glow_color = (int(color[0]*0.4), int(color[1]*0.4), int(color[2]*0.4))
    cv2.polylines(img, [pts], isClosed=False, color=glow_color, thickness=6, lineType=cv2.LINE_AA)
    
    # Draw core waveform
    cv2.polylines(img, [pts], isClosed=False, color=color, thickness=2, lineType=cv2.LINE_AA)


def generate_video(sample_rate, channels, out_video: str, file_title: str):
    print("[*] Generating visual frames. This might take a moment...")
    
    # Configuration for slicing
    samples_per_frame = max(1, sample_rate // FPS)
    window_size = int(sample_rate * 0.05) # 50ms viewing window
    total_samples = max(len(ch) for ch in channels)
    total_frames = math.ceil(total_samples / samples_per_frame)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(out_video, fourcc, FPS, RESOLUTION)
    if not writer.isOpened():
        print(f"Error: Failed to initialize video writer for '{out_video}'.")
        sys.exit(1)

    quad_labels = [
        "Paula > Samples > Ch1 (Left / Low)",
        "Paula > Samples > Ch2 (Right / Low)",
        "Paula > Samples > Ch3 (Right / High)",
        "Paula > Samples > Ch4 (Left / High)",
    ]
    quad_colors = list(COLORS.values())
    quad_offsets = [(0, 0), (QUAD_W, 0), (QUAD_W, QUAD_H), (0, QUAD_H)]

    for frame in range(total_frames):
        start_idx = frame * samples_per_frame
        end_idx = start_idx + window_size

        # Create Background
        img = np.full((RESOLUTION[1], RESOLUTION[0], 3), BG_COLOR, dtype=np.uint8)

        # Draw Grid Lines
        cv2.line(img, (QUAD_W, 0), (QUAD_W, RESOLUTION[1] - INFO_BAR_HEIGHT), GRID_COLOR, 2)
        cv2.line(img, (0, QUAD_H), (RESOLUTION[0], QUAD_H), GRID_COLOR, 2)
        cv2.line(img, (0, RESOLUTION[1] - INFO_BAR_HEIGHT), (RESOLUTION[0], RESOLUTION[1] - INFO_BAR_HEIGHT), GRID_COLOR, 2)

        # Draw Bottom Info Bar
        cv2.rectangle(img, (0, RESOLUTION[1] - INFO_BAR_HEIGHT), (RESOLUTION[0], RESOLUTION[1]), BAR_COLOR, -1)
        cv2.putText(img, "Program: PyTrackerVis", (20, RESOLUTION[1] - 30), 
                    cv2.FONT_HERSHEY_DUPLEX, 1.0, TEXT_COLOR, 1, cv2.LINE_AA)
        cv2.putText(img, "File: MOD", (20, RESOLUTION[1] - 60), 
                    cv2.FONT_HERSHEY_DUPLEX, 0.6, (180, 180, 180), 1, cv2.LINE_AA)
        
        # Draw Title right-aligned
        text_size = cv2.getTextSize(file_title, cv2.FONT_HERSHEY_DUPLEX, 1.2, 2)[0]
        title_x = RESOLUTION[0] - text_size[0] - 30
        cv2.putText(img, file_title, (title_x, RESOLUTION[1] - 30), 
                    cv2.FONT_HERSHEY_DUPLEX, 1.2, TEXT_COLOR, 2, cv2.LINE_AA)

        # Draw Waveforms
        for i in range(4):
            ch_data = channels[i]
            if start_idx < len(ch_data):
                slice_data = ch_data[start_idx : min(end_idx, len(ch_data))]
                # Pad if we reach the end
                if len(slice_data) < window_size:
                    slice_data = np.pad(slice_data, (0, window_size - len(slice_data)))
            else:
                slice_data = np.zeros(window_size)
            
            draw_waveform(img, quad_offsets[i][0], quad_offsets[i][1], 
                          slice_data, quad_colors[i], quad_labels[i])

        writer.write(img)
        
        if frame % 300 == 0 and frame > 0:
            print(f"    -> Rendered {frame} / {total_frames} frames ({(frame/total_frames)*100:.1f}%)")

    writer.release()
    print("[*] Video frames rendered successfully.")


def mux_av(video_file: str, audio_file: str, final_output: str):
    print(f"[*] Muxing audio and video into '{final_output}'...")
    cmd = [
        "ffmpeg", "-y",
        "-i", video_file,
        "-i", audio_file,
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k",
        "-shortest", final_output
    ]
    result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if result.returncode != 0:
        print("Error: ffmpeg failed to mux audio and video. Check that both temporary files are valid.")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="PyTrackerVis - Visualize 4-channel MOD files.")
    parser.add_argument("input", help="Input .mod or audio file.")
    parser.add_argument("-o", "--output", default="output.mp4", help="Output MP4 video file name.")
    parser.add_argument("-t", "--title", help="Override the display title (defaults to filename).")
    parser.add_argument("--stems", nargs=4, metavar=('CH1', 'CH2', 'CH3', 'CH4'),
                        help="Optional: Provide 4 separate WAV stems to bypass stereo splitting.")
    
    args = parser.parse_args()
    check_dependencies()

    temp_wav_file = tempfile.NamedTemporaryFile(mode='w', suffix='.wav', delete=False)
    temp_wav = temp_wav_file.name
    temp_wav_file.close()

    temp_vid_file = tempfile.NamedTemporaryFile(mode='w', suffix='.mp4', delete=False)
    temp_vid = temp_vid_file.name
    temp_vid_file.close()

    try:
        if args.stems:
            # User provided 4 independent tracks
            sample_rate, channels = load_stems(args.stems)
            # Make a stereo mix for the final video output
            stereo_mix = np.column_stack(((channels[0] + channels[3]) / 2, (channels[1] + channels[2]) / 2))
            wavfile.write(temp_wav, sample_rate, stereo_mix)
        else:
            # Process normal MOD file
            extract_audio(args.input, temp_wav)
            sample_rate, channels = split_pseudo_channels(temp_wav)

        title = args.title if args.title else Path(args.input).stem
        
        generate_video(sample_rate, channels, temp_vid, title)
        mux_av(temp_vid, temp_wav, args.output)

        print(f"\n[+] Success! Visualizer saved to '{args.output}'")

    finally:
        # Cleanup temporary files
        print("[*] Cleaning up temporary files...")
        if os.path.exists(temp_wav): os.remove(temp_wav)
        if os.path.exists(temp_vid): os.remove(temp_vid)


if __name__ == "__main__":
    main()