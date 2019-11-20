"""
 * time: 20/11/2019
 * data: selectedRID.xlsx
"""

import numpy as np
import pandas as pd
from sklearn import tree
from sklearn import svm
from sklearn import tree
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn import metrics
import pdb
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--setting", type=str, default="regression") # regression, classification
parser.add_argument("--method", type=str, default="tree") # tree, svm
parser.add_argument("--dataset", type=str, default="ALL") # BL,ML12,ML36,ALL
parser.add_argument("--evaluation", type=str, default="acc") # acc,rmse,cc
parser.add_argument("--scoretype", type=str, default="CDRSB") # CDRSB,ADAS11,ADAS13,MMSE
parser.add_argument("--training_rate", type=float, default=0.5) # 0.5, 0.2, 0.1
opt = parser.parse_args()

"""
 * When you use command line
 * Please write your command like this:
 * python analysis.py --setting classification --method svm --dataset BL --evaluation acc --scoretype CDRSB --training_rate 0.5
"""

def load_data(fp="./selectedRID.xlsx"):
	raw = pd.read_excel(fp)
	# raw.dropna(axis=0, how='any', inplace=True)
	# pdb.set_trace()
	raw.fillna(axis=0, inplace=True, method='bfill')
	labels = raw[opt.scoretype].values


	other_label = raw['VISCODE'].values
	raw.drop(['CDRSB','ADAS11','ADAS13','MMSE'],axis=1)
	data = raw.values
	data = np.delete(data, 2, axis=1)
	data = np.delete(data, 1, axis=1)
	data = np.delete(data, 0, axis=1)

	data = normalize(data, axis=0, norm='max')

	features = data
	return features,labels,other_label

def main():
	# load data from excel
	features,labels,other_label = load_data()
	train_data, test_data, train_labels, test_labels = train_test_split(features,labels,
		test_size=1-opt.training_rate,random_state=0)

	# load model
	if(opt.setting == "classification"):
		if(opt.method == "tree"):
			model = tree.DecisionTreeClassifier()
		else:
			model = svm.SVC(gamma= 'scale')
	elif(opt.setting == "regression"):
		if(opt.method == "tree"):
			model = DecisionTreeRegressor(max_depth = None)
		else: 
			model = SVR(kernel='rbf')

	#training
	model.fit(train_data, train_labels.astype(int))
	#predict
	if(opt.setting == "classification"):
		clf = []
		clf.append(model)


	# testing
	if(opt.setting == "classification"):
		pred = []
		pred.append(clf[0].predict(test_data))
		pred = np.array(pred).T
	elif(opt.setting == "regression"):
		pred = model.predict(test_data)

	# select label data
	label = []
	l1 = []
	l2 = []
	if(opt.dataset == "BL"):
		x = 0
	elif (opt.dataset == "ML12"):
		x = 1
	elif (opt.dataset == "ML36"):
		x = 2
	else : 
		x = -1  # all selected
	if(x != -1):
		for i in range(len(test_labels)):
			if(other_label[i] == x):
				l1.append(int(test_labels[i]))
				l2.append(int(pred[i]))
	else :
		l1 = test_labels
		l2 = pred

	print(l1)
	print("--------------------------------")
	print(l2)
	# get score
	# choose evaluation model
	if(opt.evaluation == "acc"):
		# score = np.sum(l1 == l2) / np.sum(np.size(l1))
		count = 0
		for i in range(len(l1)):
			if(l1[i] == l2[i]):
				count += 1
		score = count / len(l1)

	elif(opt.evaluation == "rmse"):
		score = np.sqrt(metrics.mean_squared_error(l1, l2))
	else :
		score = np.corrcoef(l1, l2)  #Correlation coefficient
	print(score)
	return score

if __name__ == '__main__':
	main()