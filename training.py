import argparse


def training():


def parsing():
	parser = argparse.ArgumentParser('vqa')
	parser.add_argument('dropout', type=float,=0.5, nargs='?')
	parser.add_argument('learning_rate', type=float, const=0.001, nargs='?')
	parser.add_argument('rec_layers', type=int, const=2)
	args = parser.parse_args()

if __name__ == '__main__':
	parsing()