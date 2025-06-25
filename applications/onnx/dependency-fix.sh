#!/bin/bash
# Fix torchvision import in basicsr/data/degradations.py using relative path
sed -i 's/from torchvision.transforms.functional_tensor import rgb_to_grayscale/from torchvision.transforms.functional import rgb_to_grayscale/' /usr/local/python/3.12.1/lib/python3.12/site-packages/basicsr/data/degradations.py
