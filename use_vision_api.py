import io
import os
import json

# Imports the Google Cloud client library
from google.cloud import vision

# Instantiates a client
client = vision.ImageAnnotatorClient()

tags_dir = 'tags'
image_dir = 'saved_anchors/image_file'
image_names = os.listdir(image_dir)
for image_name in image_names:

    image_path = os.path.abspath(os.path.join(image_dir, image_name))

    # Loads the image into memory
    with io.open(image_path, 'rb') as image_file:
        content = image_file.read()

    image = vision.Image(content=content)

    # Performs label detection on the image file
    response = client.label_detection(image=image)
    labels = response.label_annotations

    tags = []
    for label in labels:
        tags.append({'tag': label.description, 'score': label.score})

    # save the tags
    prefix = image_name.split('.')[0]
    tag_filename = f"tags_{prefix}.json"
    tag_filepath = os.path.join(tags_dir, tag_filename)
    with open(tag_filepath, 'w') as tagfile:
        json.dump(tags, tagfile)
    # break
