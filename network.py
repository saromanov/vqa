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

	def train(self, modelname='GRU'):
		model = Sequential()
		if modelname == 'GRU':
			model.add(GRU(output_dim=self.hid_dim, return_sequence=True, input_shape=(maxvalue, self.word_dim)))
		if modelname == 'LSTM':
			model.add(LSTM(output_dim=self.hid_dim, return_sequence=True, input_shape=(maxvalue, self.word_dim)))
		else:
			raise Exception("Model name is not found")
		model.add(Activation('sigmoid'))
		model.compile(loss='categorical_crossentropy', optimizer='adadelta')