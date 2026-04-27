"""
Text prompts for medical image segmentation tasks.
Used by all dataset classes to generate text-guided segmentation targets.
"""

TEXT_PROMPTS = {
    "acdc": {
        "left_ventricle": "Segment the left ventricle of the heart in cardiac MRI.",
        "right_ventricle": "Segment the right ventricle of the heart in cardiac MRI.",
        "myocardium": "Segment the myocardium in cardiac MRI.",
    },
    "spleen": {
        "spleen": "Segment the spleen in CT scan.",
    },
    "prostate": {
        "prostate": "Segment the prostate gland in ultrasound imaging.",
    },
    "cvc": {
        "polyp": "Segment the colorectal polyp in endoscopy video.",
    },
}
