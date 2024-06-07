IMAGE_FILENAMES = ['burger.jpg', 'cat.jpg']

import cv2
import math

DESIRED_HEIGHT = 1000
DESIRED_WIDTH = 1000

# def resize_and_show(image):
#   h, w = image.shape[:2]
#   if h < w:
#     img = cv2.resize(image, (DESIRED_WIDTH, math.floor(h/(w/DESIRED_WIDTH))))
#   else:
#     img = cv2.resize(image, (math.floor(w/(h/DESIRED_HEIGHT)), DESIRED_HEIGHT))
# #   cv2_imshow(img)
#   cv2.imshow("test", img)
#   cv2.waitKey(0)


# # Preview the images.

# images = {name: cv2.imread(name) for name in IMAGE_FILENAMES}
# for name, image in images.items():
#   print(name)
#   resize_and_show(image)




# STEP 1: Import the necessary modules. 모듈가져오기
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python.components import processors
from mediapipe.tasks.python import vision

# STEP 2: Create an ImageClassifier object. 추론기 만들기
base_options = python.BaseOptions(model_asset_path='models\\efficientnet_lite0.tflite')
options = vision.ImageClassifierOptions(
    base_options=base_options, max_results=1)
classifier = vision.ImageClassifier.create_from_options(options)


# STEP 3: Load the input image. 추론할 데이터 가져오고
image = mp.Image.create_from_file(IMAGE_FILENAMES[1])

# STEP 4: Classify the input image. 추론된 결과
classification_result = classifier.classify(image)

# STEP 5: Process the classification result. In this case, visualize it. 어떻게 보여줄지 
top_category = classification_result.classifications[0].categories[0]
result = f"{top_category.category_name} = ({top_category.score:.2f})"

print(result)
# display_batch_of_images(images, predictions)