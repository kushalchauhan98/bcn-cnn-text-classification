"""
Prediction script for BCN model

Usage:
	python3 predict.py /path/to/model.tar.gz /path/to/test.csv /path/to/submission.csv
"""

import sys
import pandas as pd
from tqdm import tqdm
from allennlp.predictors import Predictor
from allennlp.common.util import import_submodules
import_submodules('mylibrary')

def predict(model_path, test_path, out_path):
	hm_test = pd.read_csv(test_path, header=None)
	print('Loading model ...')
	p = Predictor.from_path(model_path, predictor_name='smile')

	print('Predicting ...')
	pred = []
	# Predicting on 10 instances at once
	for i in tqdm(range(0, len(hm_test), 10)):
		pred += p.predict(hm_test[i:10+i])
	
	# Flattening pred into a 1D array
	pred = [pred[i]['label'] for i in range(len(pred))]
	
	submit = hm_test.copy()
	
	# Replacing the cleaned_hm column with predicted labels
	submit[1] = pred
	
	submit.columns = ['hmid', 'predicted_category']
	submit.to_csv(out_path, index=False)
	
	
if __name__ == '__main__':
	if len(sys.argv) != 4:
		print('Usage:\n\tpython3 predict.py /path/to/model.tar.gz /path/to/test.csv /path/to/submission.csv')
	else:
		predict(sys.argv[1], sys.argv[2], sys.argv[3])
