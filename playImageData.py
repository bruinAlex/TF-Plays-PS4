import numpy as np 
import cv2
import argparse


file_name = r"data/training_data.npy"
train_data = np.load(file_name, allow_pickle=True)

def main(start, stop):
	# # Loop for checking training data
	# # Shows the frame and prints the input
	for data in train_data[start:stop]:
		img = data[0]
		img = cv2.resize(img, (800, 600))
		cv2.imshow('test', img)
		cv2.resizeWindow('test', (800, 600))
		if cv2.waitKey(25) & 0xFF == ord('q'):
			cv2.destroyAllWindows()
			break

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Input the start and last image numbers to play')
	parser.add_argument('start_num', type=int, help='First image number')
	parser.add_argument('end_num', type=int, help='Last image number')
	args = parser.parse_args()
	main(args.start_num, args.end_num)