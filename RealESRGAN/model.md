## Downloading the Model File

To download the model file using Git LFS (Large File Storage), follow these steps:

### Prerequisites

Ensure Git and Git LFS are installed. If not, install them using:

```bash
sudo apt update
sudo apt install git git-lfs
```

### Steps to Download

1. **Initialize Git LFS:**

```bash
git lfs install
```

2. **Download the LFS Tracked Files:**
   From within the already cloned repository:

```bash
git lfs pull
```

### Notes

* Run the above commands from within the repository folder where the `.gitattributes` and Git LFS pointers are already present.
* These steps will download the large model files tracked by Git LFS into your working directory.