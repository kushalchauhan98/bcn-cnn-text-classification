"""
Script for training BCN model
Usage:
	python3 train.py /path/to/config.jsonnet /path/to/model/folder
"""

import sys
from allennlp.commands import train as train_util
from allennlp.common.util import import_submodules
import_submodules('mylibrary')

if __name__ == '__main__':
	if len(sys.argv) != 3:
		print('Usage:\n\tpython3 train.py /path/to/config.jsonnet /path/to/model/folder')
	else:
		train_util.train_model_from_file(sys.argv[1], sys.argv[2])
