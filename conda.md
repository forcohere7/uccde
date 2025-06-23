pip install -q basicsr facexlib gfpgan numpy opencv-python Pillow torch torchvision tqdm realesrgan natsort scipy scikit-image ffmpeg-python

conda install -y -c conda-forge ffmpeg libglib

conda config --add channels defaults

conda env export > environment.yml