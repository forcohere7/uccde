import gradio as gr
import os
import subprocess
import shutil
import tempfile

def validate_directories(input_dir, output_dir):
    if not input_dir or not output_dir:
        return False, "Input and output directories must be specified."
    if not os.path.exists(input_dir):
        return False, f"Input directory '{input_dir}' does not exist."
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    if os.path.abspath(input_dir) == os.path.abspath(output_dir):
        return False, "Input and output directories must be different."
    return True, ""

def run_pipeline(image_file, enable_ip, enable_realesrgan):
    if not enable_ip and not enable_realesrgan:
        return "Error: At least one enhancement option must be selected.", None, None

    input_dir = os.path.abspath("./input")
    output_dir = os.path.abspath("./output")
    temp_dir = tempfile.mkdtemp()

    try:
        os.makedirs(input_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)
        os.chmod(input_dir, 0o755)
        os.chmod(output_dir, 0o755)
        os.chmod(temp_dir, 0o755)

        input_image_path = os.path.join(input_dir, os.path.basename(image_file.name))
        shutil.copy(image_file.name, input_image_path)

        is_valid, error_message = validate_directories(input_dir, output_dir)
        if not is_valid:
            return error_message, None, None

        if enable_ip and enable_realesrgan:
            result = subprocess.run([
                "docker-compose", "run", "--rm",
                "-v", f"{input_dir}:/app/input",
                "-v", f"{temp_dir}:/app/output",
                "image-processing", "python", "ip_inference.py",
                "--input_dir", "/app/input", "--output_dir", "/app/output"
            ], capture_output=True, text=True)
            if result.returncode != 0:
                return f"Color correction failed:\n{result.stderr}", None, None

            if not os.listdir(temp_dir):
                return "No files generated for super-resolution.", None, None

            result = subprocess.run([
                "docker-compose", "run", "--rm",
                "-v", f"{temp_dir}:/app/input",
                "-v", f"{output_dir}:/app/output",
                "real-esrgan", "/app/input", "/app/output"
            ], capture_output=True, text=True)
            if result.returncode != 0:
                return f"Super-resolution failed:\n{result.stderr}", None, None

        elif enable_ip:
            result = subprocess.run([
                "docker-compose", "run", "--rm",
                "-v", f"{input_dir}:/app/input",
                "-v", f"{output_dir}:/app/output",
                "image-processing", "python", "ip_inference.py",
                "--input_dir", "/app/input", "--output_dir", "/app/output"
            ], capture_output=True, text=True)
            if result.returncode != 0:
                return f"Color correction failed:\n{result.stderr}", None, None

        elif enable_realesrgan:
            if not os.listdir(input_dir):
                return "No input files for super-resolution.", None, None
            result = subprocess.run([
                "docker-compose", "run", "--rm",
                "-v", f"{input_dir}:/app/input",
                "-v", f"{output_dir}:/app/output",
                "real-esrgan", "/app/input", "/app/output"
            ], capture_output=True, text=True)
            if result.returncode != 0:
                return f"Super-resolution failed:\n{result.stderr}", None, None

        output_files = [f for f in os.listdir(output_dir) if os.path.isfile(os.path.join(output_dir, f))]
        if not output_files:
            return "No output files generated.", None, None

        output_file = os.path.join(output_dir, output_files[0])
        return "Enhancement completed successfully!", input_image_path, output_file

    except Exception as e:
        return f"Error: {str(e)}", None, None
    finally:
        if os.path.exists(temp_dir) and temp_dir != "/" and temp_dir != "/tmp":
            shutil.rmtree(temp_dir, ignore_errors=True)

def launch_gradio():
    with gr.Blocks() as demo:
        with gr.Column():
            gr.Markdown("# Underwater Image Enhancement")
            gr.Markdown("Enhance underwater images with color correction and super-resolution")

        with gr.Row():
            with gr.Column(scale=2):
                image_upload = gr.File(
                    label="Upload Image", 
                    file_types=["image"], 
                    file_count="single"
                )
                enable_ip = gr.Checkbox(label="Color Correction", value=True)
                enable_realesrgan = gr.Checkbox(label="Super-Resolution", value=True)
                run_button = gr.Button("Enhance Image")
                output_text = gr.Textbox(
                    label="Status", 
                    lines=3, 
                    interactive=False,
                    placeholder="Upload an image and click enhance to see results..."
                )
            with gr.Column(scale=3):
                input_image_display = gr.Image(label="Original", type="filepath")
                output_image_display = gr.Image(label="Enhanced", type="filepath")

        run_button.click(
            fn=run_pipeline,
            inputs=[image_upload, enable_ip, enable_realesrgan],
            outputs=[output_text, input_image_display, output_image_display]
        )

    demo.launch(share=False, server_name="0.0.0.0", server_port=7860)

if __name__ == "__main__":
    launch_gradio()