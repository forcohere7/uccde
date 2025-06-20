import cv2
import ffmpeg
import os
from .colour_correction import ColourCorrection
from .image_enhancement import image_enhancement
from .config import CONFIG

def process_video_frame(frame, colour_corrector):
    """Processes a single video frame."""
    corrected_frame = colour_corrector.process(frame)
    enhanced_frame = image_enhancement(corrected_frame)
    return cv2.cvtColor(enhanced_frame, cv2.COLOR_RGB2BGR)

def process_video(file, input_dir, output_dir):
    """Processes a single video and preserves original audio."""
    file_path = os.path.join(input_dir, file)
    prefix = file.split('.')[0]
    print(f'Processing video: {file}')
    
    # Open video file
    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        print(f"Could not open video: {file}")
        return
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or CONFIG['output_video_fps']
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Define output video (temporary, without audio)
    temp_output_path = os.path.join(output_dir, CONFIG['temp_video_path'])
    final_output_path = os.path.join(output_dir, f'{prefix}_ColourCorrected.mp4')
    fourcc = cv2.VideoWriter_fourcc(*CONFIG['output_video_codec'])
    out = cv2.VideoWriter(temp_output_path, fourcc, fps, (width, height))
    
    if not out.isOpened():
        print(f"Could not create output video: {temp_output_path}")
        cap.release()
        return
    
    # Initialize colour corrector
    colour_corrector = ColourCorrection()
    
    # Process frames
    print(f"Processing {frame_count} frames...")
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        print(f"Processing frame {frame_idx + 1}/{frame_count}")
        processed_frame = process_video_frame(frame, colour_corrector)
        out.write(processed_frame)
        frame_idx += 1
    
    # Release resources
    cap.release()
    out.release()
    
    # Merge original audio with processed video using ffmpeg
    try:
        print(f"Merging original audio into: {final_output_path}")
        input_video = ffmpeg.input(temp_output_path)
        input_audio = ffmpeg.input(file_path).audio
        output = ffmpeg.output(input_video.video, input_audio, final_output_path, vcodec='copy', acodec='copy', strict='experimental')
        ffmpeg.run(output, overwrite_output=True)
        print(f"Completed processing video with audio: {file}")
        
        # Clean up temporary file
        if os.path.exists(temp_output_path):
            os.remove(temp_output_path)
    except ffmpeg.Error as e:
        print(f"Error merging audio: {e.stderr.decode()}")
        return