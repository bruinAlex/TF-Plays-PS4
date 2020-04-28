import numpy as np 
import cv2
import argparse


file_name = r"data/training_data.npy"
train_data = np.load(file_name, allow_pickle=True)

def main(sample):
	img = train_data[sample, 0]
	# choice = data[1]
	img = cv2.resize(img, (800, 600))
	cv2.imshow('test', img)
	cv2.resizeWindow('test', (800, 600))
	cv2.waitKey(0)
	# print(choice)
	if cv2.waitKey(25) & 0xFF == ord('q'):
		cv2.destroyAllWindows()


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Image number to display')
	parser.add_argument('img_number', type=int, help='Input image number to display')
	args = parser.parse_args()
	main(args.img_number)