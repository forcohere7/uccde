import subprocess

def run_realesrgan_inference():
    command = [
        "python", "inference_realesrgan.py",
        "-n", "RealESRGAN_x4plus",
        "-i", "input",
        "-o", "output",
        "--model_path", "model/net_g_5000.pth",
        "--outscale", "4",
        "--suffix", "out",
        "--fp32",
    ]

    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        print("Output:\n", result.stdout)
        print("Errors (if any):\n", result.stderr)
    except subprocess.CalledProcessError as e:
        print("Inference failed with error:\n", e.stderr)

if __name__ == "__main__":
    run_realesrgan_inference()
