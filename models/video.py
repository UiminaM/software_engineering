from transformers import pipeline
from transformers import AutoImageProcessor, AutoModelForVideoClassification

processor = AutoImageProcessor.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics")
model = AutoModelForVideoClassification.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics")

pipe = pipeline("video-classification", model=model, processor=processor)
result = pipe("video.mp4")

print(result)
