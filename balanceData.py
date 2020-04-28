import numpy as np 
import pandas as pd
from collections import Counter
from random import shuffle
import cv2

file_name = r"data/training_data.npy"
train_data = np.load(file_name, allow_pickle=True)


df = pd.DataFrame(train_data)
print(f"Initial data length: {len(train_data)}")


# # How unbalanced is our data?
# print(Counter(df[1].apply(str)))


click = []
no_click = []

shuffle(train_data)


for data in train_data:
	img = data[0]
	choice = data[1]

	if choice == [1]:
		click.append([img, choice])
	else:
		no_click.append([img, choice])


# Balance the data
# This can be skipped in lieu of another balancing method
# or img augmentation for more positives
click = click[:len(no_click)]
no_click = no_click[:len(click)]


final_data = click + no_click


shuffle(final_data)


print("Done shuffling!")
print(f"Final data length: {len(final_data)}")
np.save("data/training_data_shuffled.npy", final_data)



# # Loop for checking training data
# # Shows the frame and prints the input
# for data in train_data:
# 	img = data[0]
# 	choice = data[1]
# 	cv2.imshow('test', img)
# 	print(choice)
# 	if cv2.waitKey(25) & 0xFF == ord('q'):
# 		cv2.destroyAllWindows()
# 		break

