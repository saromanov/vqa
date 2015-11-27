from kears.models  import Sequential
from keras.layers.recurrent import GRU, LSTM
from keras.layers.core import Dropout, Activation, Dense


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