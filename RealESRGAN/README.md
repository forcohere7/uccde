python src/inference_realesrgan_image.py -n RealESRGAN_x4plus -i input -o output --model_path model/net_g_5000.pth --outscale 4 --tile 400

python src/inference_realesrgan_video.py -i video/test1.mp4 -o output --model_path model/net_g_5000.pth --outscale 4 --tile 400


docker run --rm -v $(pwd)/input:/app/input -v $(pwd)/output:/app/output real-esrgan /app/input
docker run --rm -v $(pwd)/video:/app/video -v $(pwd)/output:/app/output real-esrgan /app/video/test1.mp4