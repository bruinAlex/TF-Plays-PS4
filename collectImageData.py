# collectImageData.py
import numpy as np
from PIL import ImageGrab # This is still using Pillow 7.1.1 in background
import cv2
import time
import locateButtons
import win32api as wapi
import os


# Make and store training data (screen shots)
file_name = r"data/training_data.npy"

if os.path.isfile(file_name):
	print("File exists, loading previous data!")
	training_data = list(np.load(file_name, allow_pickle=True))
else:
	print("File does not exist, starting fresh")
	training_data = []


def mouse_clicks_to_output():
	"""
	Return a 1 when a mouse click is detected, else return a 0
	"""
	# Mouse left click is 0x01
	# [0x01]
	output = 0 # default output is not-clicked

	if wapi.GetAsyncKeyState(0x01):
		output = 1

	return output


def main():
	last_time = time.time()

	# Grabs entire screen (not great on high res monitor)
	while(True):
		screen = np.array(ImageGrab.grab(bbox=(0, 0, 800, 640))) # bbox = (left, upper, right, lower)

		# Convert screen to grayscale (1/3rd the size of RGB)
		screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)

		# Resize data smaller
		screen = cv2.resize(screen, (80, 60))

		# Check for mouse click
		output = mouse_clicks_to_output()

		# Append screen output to our training data
		training_data.append([screen, output])

		# # Time how long between frames captured
		# print(f"Loop took {time.time() - last_time} seconds")
		# last_time = time.time()

		# Every 500 caps, save to training data
		if len(training_data) % 500 == 0:
			np.save(file_name, training_data)
			print(f"Updating training data. Total of {len(training_data)} frames")
			


if __name__ == '__main__':
	main()