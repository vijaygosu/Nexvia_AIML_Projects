"""Shared constants. Keep this file dependency-free (stdlib only)."""

DISCLAIMER = (
    "FOR EDUCATION ONLY - NOT FOR CLINICAL USE. This is a research prototype that has not been "
    "clinically validated. Do not use its output for diagnosis or treatment decisions."
)

# ChestX-ray8 subset (default) and the full ChestX-ray14 label set.
LABELS_8 = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
    "Mass", "Nodule", "Pneumonia", "Pneumothorax",
]
LABELS_14 = LABELS_8 + [
    "Consolidation", "Edema", "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia",
]

IMG_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
