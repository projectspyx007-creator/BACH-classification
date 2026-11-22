
BACH RESEARCH WEBSITE PACKAGE

This package contains:
- Original hybrid global-local model
- ImageNet EfficientNetB0 feature extractor
- Patient-grouped dataset manifest
- Original 88-patch coordinates
- Final internal-test evaluation
- Training logs
- Attention visualization examples
- Sample processed images
- Model architecture information

IMPORTANT:
The original model and feature extraction pipeline must be preserved.

Global input:
Processed 224x224 RGB JPEG -> EfficientNetB0 -> 1280 features

Local input:
Original TIFF -> 88 overlapping 224x224 patches
-> EfficientNetB0 -> (88, 1280)

Hybrid model:
[Global features, Local features] -> 4-class prediction

This is a research prototype, not a clinical diagnostic system.
