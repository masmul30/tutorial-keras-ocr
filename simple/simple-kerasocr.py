import matplotlib.pyplot as plt
import keras_ocr
import os

pipeline = keras_ocr.pipeline.Pipeline()
image = keras_ocr.tools.read(‘/content/uploads/EUBanana-500x112.jpg’)
predictions = pipeline.recognize([image])[0]
fig, ax = plt.subplots()
drawn=keras_ocr.tools.drawAnnotations(image=image, predictions=predictions, ax=ax)
plt.savefig(‘test.jpg’)
print(‘Predicted:’, [text for text, box in predictions])
with open(‘results.txt’, ‘w+’) as f:
f.write(str([text for text, box in predictions]))
