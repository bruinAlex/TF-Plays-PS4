import numpy as np
from alexnetModified import alexnet
import tensorflow as tf

WIDTH = 80
HEIGHT = 60
LR = 1e-3
EPOCHS = 12
INFO = "upsampled"
MODEL_NAME = "cod_mw_gw_realism-{}-{}-{}-{}-epochs.model".format(INFO, LR, 'alexnet', EPOCHS)

model = alexnet()

train_data = np.load(r"data/training_data_upsampled.npy", allow_pickle=True)

# 80% split on shuffled data
SPLIT_INDEX = 12300
train = train_data[:SPLIT_INDEX]
test = train_data[SPLIT_INDEX:]

# Reshape our training data
train_X = np.array([i[0] for i in train]).reshape(-1, WIDTH, HEIGHT, 1)
train_y = np.array([i[1] for i in train])

# Reshape our test data
test_X = np.array([i[0] for i in test]).reshape(-1, WIDTH, HEIGHT, 1)
test_y = np.array([i[1] for i in test])

model.compile(
	optimizer='adam',
	loss='sparse_categorical_crossentropy',
	metrics=[tf.keras.metrics.CategoricalCrossentropy()]
	)

model.fit(
	x=train_X,
	y=train_y,
	epochs=EPOCHS,
	verbose=1,
	validation_data=(test_X, test_y),
	)

model.save(MODEL_NAME)