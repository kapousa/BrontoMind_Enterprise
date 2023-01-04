import yolov5

# model
model = yolov5.load('yolov5s')

# image
img = 'https://github.com/ultralytics/yolov5/raw/master/data/images/zidane.jpg'

# inference
results = model(img)

# inference with larger input size
results = model(img, size=1280)

# inference with test time augmentation
results = model(img, augment=True)

# show results
results.show()

# save results
results.save(save_dir='results/')