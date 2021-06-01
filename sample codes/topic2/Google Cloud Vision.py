# gcloud auth list
# gcloud config set project <PROJECT_ID>
# virtualenv my_env
# source my_env/bin/activate
# gcloud services enable vision.googleapis.com
# export PROJECT_ID=$(gcloud config get-value core/project)
# gcloud iam service-accounts create my-vision-sa \
  --display-name "my vision service account"
# gcloud iam service-accounts keys create ~/key.json \
  --iam-account my-vision-sa@${PROJECT_ID}.iam.gserviceaccount.com
# export GOOGLE_APPLICATION_CREDENTIALS=~/key.json
# pip3 install -U pip google-cloud-vision

# Label Detection

from __future__ import print_function
from google.cloud import vision

image_uri = 'gs://cloud-samples-data/vision/using_curl/shanghai.jpeg'

client = vision.ImageAnnotatorClient()
image = vision.Image()
image.source.image_uri = image_uri

response = client.label_detection(image=image)

print('Labels (and confidence score):')
print('=' * 30)
for label in response.label_annotations:
    print(label.description, '(%.2f%%)' % (label.score*100.))


# Text Detection

from __future__ import print_function
from google.cloud import vision

image_uri = 'gs://cloud-vision-codelab/otter_crossing.jpg'

client = vision.ImageAnnotatorClient()
image = vision.Image()
image.source.image_uri = image_uri

response = client.text_detection(image=image)

for text in response.text_annotations:
    print('=' * 30)
    print(text.description)
    vertices = ['(%s,%s)' % (v.x, v.y) for v in text.bounding_poly.vertices]
    print('bounds:', ",".join(vertices


# Landmark Detection

from __future__ import print_function
from google.cloud import vision

image_uri = 'gs://cloud-vision-codelab/eiffel_tower.jpg'

client = vision.ImageAnnotatorClient()
image = vision.Image()
image.source.image_uri = image_uri

response = client.landmark_detection(image=image)

for landmark in response.landmark_annotations:
    print('=' * 30)
    print(landmark)


# Emotional Face Detection

from __future__ import print_function
from google.cloud import vision

uri_base = 'gs://cloud-vision-codelab'
pics = ('face_surprise.jpg', 'face_no_surprise.png')

client = vision.ImageAnnotatorClient()
image = vision.Image()

for pic in pics:
    image.source.image_uri = '%s/%s' % (uri_base, pic)
    response = client.face_detection(image=image)

    print('=' * 30)
    print('File:', pic)
    for face in response.face_annotations:
        likelihood = vision.Likelihood(face.surprise_likelihood)
        vertices = ['(%s,%s)' % (v.x, v.y) for v in face.bounding_poly.vertices]
        print('Face surprised:', likelihood.name)
        print('Face bounds:', ",".join(vertices))