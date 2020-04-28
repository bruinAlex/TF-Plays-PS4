import numpy as np 
import cv2
import time
from PIL import ImageGrab # This is still using Pillow 7.1.1 in background
from mouseCheck import mouse_check
import os
from tensorflow import keras
from locateButtons import click_play_button_off, click_stop_button_off, detect_play_button_on, detect_stop_button_on


WIDTH = 80
HEIGHT = 60
LR = 1e-3
EPOCHS = 12
INFO = "upsampled_test"
MODEL_NAME = "cod_mw_gw_realism-{}-{}-{}-{}-epochs.model".format(INFO, LR, 'alexnet', EPOCHS)


# Load our trained model
model = keras.models.load_model(MODEL_NAME)

def main(debug=True):
	# Countdown before model runs
	for i in list(range(1))[::-1]:
		print(i + 1)
		time.sleep(1)

	last_time = time.time()

	# Initial macro on/off state
	# This is used to prevent multiple consecutive calls to pyautogui which is slow
	if detect_stop_button_on():
		macro_on_flag = False
	else:
		macro_on_flag = True

	# Grabs entire screen (not great on high res monitor)
	while(True):
		# Set up consecutive frames for checking
		consecutive_frames = 0
		weak_max_click_index = None

		screen = np.array(ImageGrab.grab(bbox=(0, 0, 800, 640))) # bbox = (left, upper, right, lower)

		# Convert screen to grayscale (1/3rd the size of RGB)
		screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)

		# Resize data smaller
		screen = cv2.resize(screen, (WIDTH, HEIGHT))


		# # Time how long between frames captured
		# print(f"Loop took {time.time() - last_time} seconds")
		# last_time = time.time()


		# Predict on the new screen
		prediction = model.predict([screen.reshape(-1, WIDTH, HEIGHT, 1)])
		click = list(np.around(prediction))
		# Due to borderline cases we only want very high results of > 85% for class 2
		# Convert click array into a list, find the index of the max value and return
		THRESHOLD = 0.7
		if max(prediction[0]) < THRESHOLD:
			max_click_index = 0
			print(f"Below threshold of {THRESHOLD}")

			# Track consecutive weak triggers
			new_weak_max_click_index = click[0].tolist().index(max(click[0]))
			if new_weak_max_click_index != weak_max_click_index:
				weak_max_click_index = new_weak_max_click_index
				consecutive_frames = 1
			else:
				consecutive_frames += 1

		else:
			max_click_index = click[0].tolist().index(max(click[0]))
		print(click, max_click_index, prediction, macro_on_flag)

		# This checks for consecutive weak triggers
		# If it hits 30
		CONSECUTIVE_THRESHOLD = 30
		if consecutive_frames == CONSECUTIVE_THRESHOLD:
			consecutive_frames = 0
			max_click_index = weak_max_click_index
			weak_max_click_index = None


		# Use prediction to trigger pyautogui
		if max_click_index == 1:
			if macro_on_flag == False:
				if detect_stop_button_on():
					click_play_button_off()
					print("Macro started!")
					macro_on_flag = True

					if debug:
						# Save a screenshot when the macro is stopped for debugging
						screenshot_file_name = r"logging_images/{}-{}.jpg".format(max_click_index, time.time())
						cv2.imwrite(screenshot_file_name, screen)
						
		elif max_click_index == 2:
			if macro_on_flag == True:
				if detect_play_button_on():
					click_stop_button_off()
					print("Macro stopped!")
					macro_on_flag = False

					if debug:
						# Save a screenshot when the macro is stopped for debugging
						screenshot_file_name = r"logging_images/{}-{}.jpg".format(max_click_index, time.time())
						cv2.imwrite(screenshot_file_name, screen)
		else:
			pass


if __name__ == '__main__':
	main()