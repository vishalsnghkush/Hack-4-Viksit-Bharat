import sys
print(f"Python version: {sys.version}")

print("Importing numpy...")
try:
    import numpy
    print("numpy ok")
except Exception as e:
    print(f"numpy failed: {e}")

print("Importing PIL...")
try:
    import PIL
    from PIL import Image
    print("PIL ok")
except Exception as e:
    print(f"PIL failed: {e}")

print("Importing cv2...")
try:
    import cv2
    print("cv2 ok")
except Exception as e:
    print(f"cv2 failed: {e}")

print("Importing torch...")
try:
    import torch
    print("torch ok")
except Exception as e:
    print(f"torch failed: {e}")

print("Importing transformers...")
try:
    from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
    print("transformers ok")
except Exception as e:
    print(f"transformers failed: {e}")
