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

	def train(self):
		model = Sequential()
		model.add(GRU(output_dim=self.hid_dim, return_sequence=False, input_shape=(maxvalue, self.word_dim)))
		model.add(Activation('sigmoid'))
		model.compile(loss='categorical_crossentropy', optimizer='adadelta')