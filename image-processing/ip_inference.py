import os
import datetime
import natsort
from multiprocessing import Pool, cpu_count
import cv2
from src import CONFIG, ColourCorrection, image_enhancement, process_video

def process_image(file, input_dir=CONFIG['input_dir'], output_dir=CONFIG['output_dir']):
    """Processes a single image."""
    file_path = os.path.join(input_dir, file)
    prefix = file.split('.')[0]
    print(f'Processing image: {file}')
    img = cv2.imread(file_path)
    if img is None:
        print(f"Could not read image: {file}")
        return
    colour_corrector = ColourCorrection()
    print("Applying color correction...")
    corrected_img = colour_corrector.process(img)
    print("Applying image enhancement...")
    final_result = image_enhancement(corrected_img)
    final_result_bgr = cv2.cvtColor(final_result, cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(output_dir, f'{prefix}_ColourCorrected.jpg'), final_result_bgr)
    print(f"Completed processing image: {file}")

def main():
    """Main function to process images and videos in the input directory."""
    os.makedirs(CONFIG['output_dir'], exist_ok=True)
    if not os.access(CONFIG['output_dir'], os.W_OK):
        print(f"No write permissions for directory {CONFIG['output_dir']}")
        exit(1)

    start_time = datetime.datetime.now()
    files = natsort.natsorted(os.listdir(CONFIG['input_dir']))
    files = [f for f in files if os.path.isfile(os.path.join(CONFIG['input_dir'], f))]

    # Separate images and videos
    image_files = [f for f in files if os.path.splitext(f)[1].lower() not in CONFIG['video_extensions']]
    video_files = [f for f in files if os.path.splitext(f)[1].lower() in CONFIG['video_extensions']]

    # Process images using multiprocessing
    if image_files:
        print(f"Processing {len(image_files)} images...")
        with Pool(processes=cpu_count()) as pool:
            pool.map(process_image, image_files)

    # Process videos sequentially
    if video_files:
        print(f"Processing {len(video_files)} videos...")
        for video_file in video_files:
            process_video(video_file)

    print(f'Total processing time: {datetime.datetime.now() - start_time}')

if __name__ == "__main__":
    main()