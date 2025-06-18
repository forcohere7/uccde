#!/bin/bash
# Fix torchvision import in basicsr/data/degradations.py
sed -i 's/from torchvision.transforms.functional_tensor import rgb_to_grayscale/from torchvision.transforms.functional import rgb_to_grayscale/' /app/.venv/lib/python3.12/site-packages/basicsr/data/degradations.py