CONFIG = {
    'block_size': 9,
    'gimfilt_radius': 50,
    'eps': 1e-3,
    'rb_compensation_flag': 0,  # 0: Compensate both Red and Blue, 1: Compensate only Red
    'video_extensions': ['.mp4', '.avi', '.mov'],  # Supported video formats
    'output_video_fps': 30,  # Default output video frame rate
    'output_video_codec': 'mp4v',  # Codec for output video
    'temp_video_path': 'temp_output.mp4',  # Temporary video file without audio
}