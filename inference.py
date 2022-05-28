import os
import glob
import cv2
import pandas as pd
import numpy as np
import shutil


def yolo_to_center_coord(img, str_):
	dh, dw, _ = img.shape
	_, x, y, w, h = map(float, str_.split())
	return dw * x, dh * y


if __name__ == "__main__":
	print('script started')

	weights_name = 'best_90.pt'
	img_size = 1024
	conf = 0.15
	iou_thres = 0.25
	images = './test_images'
	name_task = 'yolo_test_images_detect'

	os.system(f'python ./yolov5/detect.py --weights ./weights/{weights_name} --img {img_size} --conf {conf} --iou-thres {iou_thres} --source {images} --name {name_task} --save-txt')
	images_path = './test_images'
	predicted_labels_path = './yolov5/runs/detect/yolo_test_images_detect/labels'
	images = sorted(glob.glob(images_path + '/*.jpg'))
	labels = sorted(glob.glob(predicted_labels_path + '/*.txt'))

	assert (len(labels) == len(images))

	for i, label in enumerate(labels):
		fl = open(label, 'r')
		data = fl.readlines()
		fl.close()
		image = cv2.imread(images[i])
		image_file_name = images[i].replace('\\', '/').split('/')[-1].replace('.jpg', '')

		center_coords = []
		for dt in data:
			center_coord = yolo_to_center_coord(image, dt)
			center_coord = np.int32(center_coord[0]), np.int32(center_coord[1])
			center_coords.append(center_coord)
		df = pd.DataFrame(center_coords)
		df = df.rename(columns={0: "x", 1: "y"})
		if not os.path.exists('./coords_center'):
			os.mkdir('./coords_center')
		df.to_csv(f'./coords_center/{image_file_name}.csv', index=False)

	REMOVE_FILES = True
	if REMOVE_FILES:
		shutil.rmtree('./yolov5/runs/detect/yolo_test_images_detect/', ignore_errors=True)
		print('dirs removed')




