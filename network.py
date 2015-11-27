from kears.models  import Sequential
from keras.layers.recurrent import GRU, LSTM
from keras.layers.core import Dropout, Activation, Dense, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D


class Recurrent:
	''' Training recurrent NN
	'''
	def __init__(self, inp_dim, hid_dim, word_dim):
		self.inp_dim = inp_dim
		self.hid_dim
		self.word_dim = word_dim

	def train(self, X, y, Xtest, ytest, modelname='GRU', batch_size=32):
		model = Sequential()
		if modelname == 'GRU':
			model.add(GRU(output_dim=self.hid_dim, return_sequence=True, input_shape=(maxvalue, self.word_dim)))
		if modelname == 'LSTM':
			model.add(LSTM(output_dim=self.hid_dim, return_sequence=True, input_shape=(maxvalue, self.word_dim)))
		else:
			raise Exception("Model name is not found")
		model.add(Dropout(0.5))
		model.add(Activation('sigmoid'))
		model.compile(loss='categorical_crossentropy', optimizer='adadelta')
		model.fit(X,y, batch_size=batch_size, nb_epoch=4, validation_data=(Xtest, ytest), show_accuracy=True)
		score, acc = model.evaluate(Xtest, ytest, batch_size=batch_size, show_accuracy=True)
		print('Score: {0}'.format(score))
		print('Accuracy: {0}'.format(acc))


class Convolutional:
	def __init__(self, dim, num_classes):
		self.dim = dim
		self.num_classes = num_classes

	def train(self, X, y, Xtest, ytest):
		model = Sequential()
		model.add(Convolution2D(32,3,3)
		model.add(Activation('relu'))
		model.add(MaxPooling2D(pool_size=(2,2)))
		model.add(Activation('relu'))
		model.add(Dropout(0.4))
		model.add(Convolution2D(64,3,3, border_mode='full'))
		model.add(Activation('relu'))
		model.add(MaxPooling2D(pool_size=(2,2)))
		model.add(Dropout(0.2))
		model.add(Flatten())
		model.add(Dense(512))
		model.add(Dropout(0.5))
		model.add(Dense(self.num_classes))
		model.add(Activation('softmax'))

		model.compile(loss='categorical_crossentropy', optimizer='adam')
		X = X.astype("float32")
		Xtest = X.astype("float32")
		X /=255
		Xtest /= 255
		model.fit(X, y)