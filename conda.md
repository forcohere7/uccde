# Saving Conda Environment with Additional Pip Dependencies

This guide walks you through installing required packages using both `pip` and `conda`, and then exporting your environment to a `environment.yml` file for reproducibility.

## Step 1: Install Python Packages using pip

Use the following command to quietly install the required Python packages:

```bash
pip install -q basicsr facexlib gfpgan numpy opencv-python Pillow torch torchvision tqdm realesrgan natsort scipy scikit-image ffmpeg-python
```

## Step 2: Install System-level Dependencies using conda

Install additional system-level libraries like `ffmpeg` and `libglib` that may not be available or ideal to install via `pip`:

```bash
conda install -y -c conda-forge ffmpeg libglib
```

## Step 3: Ensure `defaults` Channel is Added

Verify the default channel is added for compatibility:

```bash
conda config --add channels defaults
```

> This step ensures conda has access to packages that are not available in `conda-forge`.

## Step 4: Export the Conda Environment

To export the entire environment including dependencies to a `environment.yml` file:

```bash
conda env export > environment.yml
```

### Notes:

* This will include both conda-installed and pip-installed packages.
* Make sure to run this command from the root of your project or preferred location to save the `environment.yml` file.

## Optional: Save to Specific Relative Paths

To save the `environment.yml` file and Jupyter notebook in your desired structure using **relative paths**:

### Save Conda Environment:

```bash
conda env export > ../environment.yml
```

(This saves it as `environment.yml` in the root directory.)