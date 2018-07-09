import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import roc_auc_score
from dataset import DataSet
import xgboost as xgb

####################################################################################################################################################  
####################################################################################################################################################
####################################################################################################################################################  
####################################################################################################################################################
#
#         General Overview
#
#   Module 0 : Specifying feature names and output file names
#   Module 1 : Preprocessing
#   Module 2 : Defining a few functions
#   Module 3 : Running the main functions
#

####################################################################################################################################################  
####################################################################################################################################################
####################################################################################################################################################  
####################################################################################################################################################
#
#   Module 0 : Specifying paths and file names
#
#   IMPORTANT: lines 338 - 340 must be customized such that the XLLCORNER
#   and YLLCORNER parameters are the coordinates of the bottom left corner
#   of the discretization in automate_data. In addion the CELLSIZE parameter
#   must also match accordingly
#

# name of text file output for probabilistic predictions of 
# each grid cell
qgis_file_in = "predictions.txt"
# raster file of probabilistic predictions
qgis_file_out = "predictions_heatmap.asc"
# specify which features to use from final.csv feature spreadsheet
selected_features = ["dist-village",
					 "dist-patrol",
					 "dist-example",
					 "dist-feature",
					 "is-names",
					 "is-highway",
					 "dem",
					 "slope"]
# specify which feature symbolizes where patrolling occurs
patrol = 'patrol'
# specify which feature symbolizes where poaching occurs
poaching = 'poaching'

####################################################################################################################################################  
####################################################################################################################################################
####################################################################################################################################################  
####################################################################################################################################################
#
#   Module 1 : Preprocessing
#

df_alldata = pd.read_excel("final.csv")

# select data without nan
df_validdata = df_alldata.dropna()

# select data with nan 
df_invaliddata = df_alldata[df_alldata.isnull().any(axis=1)]

# select labeled data
df_knowndata = df_validdata[(df_validdata[patrol]>0)]

# select unlabeled data
df_unknowndata = df_validdata[(df_validdata[patrol]==0)]

# obtain positive data, replace 'Poaching-17' and others with feature 
# names that specify existence of previous poaching 
df_allpositive = df_knowndata[(df_knowndata[poaching] != 0)]
# if multiple features format as follows
# df_allpositive = df_knowndata[(df_knowndata[poaching1] != 0) | 
#							 (df_knowndata[poaching2] != 0) | 
#							 (df_knowndata[poaching3] != 0)]

# obtain negative data, replace 'Poaching-17' and others with feature 
# names that specify existence of previous poaching 
df_allnegative = df_knowndata[(df_knowndata[poaching] == 0)]	
# if multiple features format as follows						 
# df_allnegative = df_knowndata[(df_knowndata[poaching1] == 0) & 
#							 (df_knowndata[poaching2] == 0) & 
#							 (df_knowndata[poaching3] == 0)]							 

df_slct_positive = df_allpositive[selected_features]

df_slct_negative = df_allnegative[selected_features]

df_slct_unlabeled = df_unknowndata[selected_features]

PositiveData = df_slct_positive.values

NegativeData = df_slct_negative.values

UnknownData = df_slct_unlabeled.values

maxvals = np.max(np.concatenate([PositiveData,NegativeData,UnknownData],axis=0),0,keepdims=True)[0:1]
maxvals[maxvals==0]=1.

PositiveData = PositiveData/maxvals
NegativeData = NegativeData/maxvals
UnknownData = UnknownData/maxvals

fold_pos_num = len(PositiveData) // FoldNum
fold_neg_num = len(NegativeData) // FoldNum

# shuffle the negative data
np.random.shuffle(NegativeData)
neg = NegativeData[:fold_neg_num]
NegativeData = NegativeData[fold_neg_num:]

# negative sampling here
sample_size = NegativeData.shape[0]
indx = np.random.randint(UnknownData.shape[0], size=sample_size)
Udata = UnknownData[indx]
#Udata1 = UnknownData[index[-sample_size:]]

NotFam = np.concatenate((Udata, NegativeData), axis=0)
neg_label = np.array([0.] * len(neg))
Fam = PositiveData

dataset = DataSet(positive=Fam, negative=NotFam, fold_num=FoldNum)

# number of folds specified for classify_familiar_trial
FoldNum = 4

####################################################################################################################################################  
####################################################################################################################################################
####################################################################################################################################################  
####################################################################################################################################################
#
#   Module 2 : Defining the main functions
#

# run cross validation and test decision tree ensembling with xgboost, dynamic negative sampling, and oversampleing
# on the given data
def classify_familiar_trial():
	print("start classify familiar trial")
	test_auc_list = []
	train_auc_list = []
	numIter = 10
	total_list = np.array([0.0,0.0,0.0,0.0,0.0])
	for iters in range(numIter):
		dataset = DataSet(positive=Fam, negative=NotFam, fold_num=FoldNum)
		for fold_id in range(FoldNum):
			dataset.update_negative(NotFam)
			y_prob_sum = 0
			for j in range(1):
				#train_data, train_label, test_data_pos, test_label_pos = dataset.get_train_neg_traintest_pos_aug(cluster_ids, cluster_ids50, index, UnknownData, UnknownDataID, fold_id, 175)
				#train_data, train_label, test_data_pos, test_label_pos = dataset.get_train_neg_traintest_pos(fold_id, 300)
				train_data, train_label, test_data_pos, test_label_pos = dataset.get_train_neg_traintest_pos_smote(fold_id, 300)
				test_data = np.concatenate((test_data_pos, neg), axis=0)
				#test_data[:,12]=0
				test_label = np.concatenate((test_label_pos, neg_label), axis=0)

				#################### Decision Tree ###################################
				# clf = BaggingClassifier(tree.DecisionTreeClassifier(criterion = "entropy"), n_estimators=1000, max_samples=0.1)
				# clf.fit(train_data, train_label)
				# unl_scores = clf.predict_proba(UnknownData)[:,1]
				# y_prob = clf.predict_proba(test_data)
				# y_prob_sum = y_prob
				# y_pred = np.argmax(y_prob_sum,1)
				######################################################################

				##################### xgboost ##################################
				D_train = xgb.DMatrix(train_data, label = train_label)
				D_test = xgb.DMatrix(test_data, label = test_label)

				param = {'max_depth': 20, 'eta': 1, 'silent': 1, 'objective': 'binary:logistic'}
				num_round = 5

				bst = xgb.train(param, D_train, num_round)

				D_unknownData = xgb.DMatrix(UnknownData)

				unl_scores = bst.predict(D_unknownData)

				###############################################################
				train_prob = bst.predict(D_train)
				train_pred = train_prob
				
				for i in range(0, len(train_pred)):
					if train_pred[i] > 0.5:
						train_pred[i] = 1.0
					else:
						train_pred[i] = 0.0

				###############################################################
				test_prob = bst.predict(D_test)
				test_pred = test_prob
				
				for i in range(0, len(test_pred)):
					if test_pred[i] > 0.5:
						test_pred[i] = 1.0
					else:
						test_pred[i] = 0.0

				###############################################################

				test_auc = roc_auc_score(y_true=test_label, y_score=test_pred)
				train_auc = roc_auc_score(y_true=train_label, y_score=train_pred)
				
				test_tp = np.sum(test_pred*test_label)
				test_fn = np.sum((1-test_pred)*test_label)
				test_fp = np.sum(test_pred*(1-test_label))
				train_tp = np.sum(train_pred*train_label)
				train_fn = np.sum((1-train_pred)*train_label)
				train_fp = np.sum(train_pred*(1-train_label))

				test_recall = test_tp/(test_tp + test_fn)
				test_per = test_tp/(test_tp + test_fp)
				train_recall = train_tp/(train_tp + train_fn)
				train_per = test_tp/(train_tp + train_fp)

				test_denominator = np.sum(test_pred)/np.float(len(test_pred))
				train_denominator = np.sum(train_pred)/np.float(len(train_pred))
				test_lnl = np.square((test_tp/(test_tp+test_fn)))/test_denominator
				train_lnl = np.square((train_tp/(train_tp+train_fn)))/train_denominator
			   
				test_f1score = 2 * test_per * test_recall / \
								(test_per + test_recall)
				train_f1score = 2 * test_per * test_recall / \
								(test_per + test_recall)

				print("test:", np.mean(test_auc), test_lnl, test_recall, \
								test_per, test_f1score)
				print("train:", np.mean(train_auc), train_lnl, train_recall, \
								train_per, train_f1score)

			test_auc_list.append([test_auc, test_lnl, test_recall, test_per, test_f1score])
			train_auc_list.append([train_auc, train_lnl, train_recall, train_per, train_f1score])

		print('mean for test: ', np.mean(test_auc_list,axis=0))
		#print('mean for train: ', np.mean(train_auc_list,axis=0))
		cur_auc_list = np.mean(test_auc_list,axis=0)
		total_list = total_list + cur_auc_list

	total_list = total_list / numIter
	print('mean over all iteration (test)', total_list)

# Generate the actual predictions by training on all the data
# qgis_file_in is the output of this function, which contains the probabilistic
# predictions as a text file
def main_poaching_predict(qgis_file_in):
	print("start main_poaching_predict")

	PositiveDataID = df_allpositive["id"].values
	NegativeDataID = df_allnegative["id"].values
	UnknownDataID = df_unknowndata["id"].values

	ALLID = list(PositiveDataID) + list(NegativeDataID) + list(UnknownDataID)
	ALLDATA = list(df_slct_positive.values) + \
			  list(df_slct_negative.values) + \
			  list(df_slct_unlabeled.values)

	# how to sampling here
	train_data, train_label = dataset.get_train_all_up(90)
	# clf = BaggingClassifier(tree.DecisionTreeClassifier(), n_estimators=1000, max_samples=0.1)
	# clf.fit(train_data, train_label)

	param = {'max_depth': 15, 'eta': 0.3, 'silent': 1, 'objective': 'binary:logistic'}
	num_round = 10


	# param = {'max_depth': 20, 'eta': 0.1, 'silent': 1, 'objective': 'binary:logistic'}
	# num_round = 10

	D_train = xgb.DMatrix(train_data, label = train_label)

	bst = xgb.train(param, D_train, num_round)


	ALLDATA = np.array(ALLDATA)

	D_ALLDATA = xgb.DMatrix(ALLDATA)

	# prediction results
	ALL_value = bst.predict(D_ALLDATA)
	ALL_scores = np.zeros(len(ALL_value))

	for i in range(0, len(ALL_value)):
		if (ALL_value[i] > 0.5):
			ALL_scores[i] = 1.0
		else:
			ALL_scores[i] = 0.0


	id_label = zip(ALLID, ALL_value)
	id_label = list(id_label)

	Invalid_ID = df_invaliddata["id"].values
	for id in Invalid_ID:
		id_label.append((id, 0.0))

	id_label = sorted(id_label, key=lambda x: x[0], reverse=False)
	# print (id_label)
	with open(qgis_file_in, 'w')as fout:
		fout.write('ID\tLabel\n')
		for idx, label in id_label:
			temp_str = str(idx) + '\t' + str(label) + '\n'
			fout.write(temp_str)

# translates output from main_poaching_predict into an ASC file
def prep_qgis(qgis_file_in, qgis_file_out):
	print("start prep_qgis")
	l_id = df_alldata["id"].values
	l_X = df_alldata["X"].values
	l_Y = df_alldata["Y"].values

	if (len(l_id) != len(l_X)) or (len(l_X) != len(l_Y)):
		print ("prep_qgis dim not match")

	ID_coordinate = dict()

	for i in range(0, len(l_id)):
		ID_coordinate[l_id[i]] = (l_X[i], l_Y[i])

	# Map configuration
	x_set = set()
	y_set = set()
	for index in ID_coordinate:
		x_set.add(ID_coordinate[index][0])
		y_set.add(ID_coordinate[index][1])
	max_x = int(max(x_set) / 200)
	max_y = int(max(y_set) / 200)

	Map = np.zeros([max_y + 1, max_x + 1])
	
	# Load target list
	id_label = {}
	num_clusters = 50

	file_in = qgis_file_in
	file_out = qgis_file_out

	# file_in = "紫貂_result_predict.txt"
	# file_out = "ASC/紫貂_predict_heatmap.asc"

	with open(file_in) as fin:
		fin.readline()
		for line in fin:
			line = line.strip().split()
			#print(line[0])
			index = int(float(line[0]))
			label = float(line[1])
			id_label[index] = label

	count = 0
	for index in ID_coordinate:
		id_x = int(ID_coordinate[index][0] / 200)
		id_y = int(ID_coordinate[index][1] / 200)
		try:
			Map[id_y, id_x] = id_label[index]
		except:
			count += 1
			# print index
			Map[id_y, id_x] = 0.0

	print("number of key error: %d" % count)

	with open(file_out, 'w') as fout:
		fout.write('NCOLS ' + str(max_x + 1) + '\n')
		fout.write('NROWS ' + str(max_y + 1) + '\n')
		#following coordinates need to be changed
		fout.write('XLLCORNER 22393800\n')
		fout.write('YLLCORNER 4810000\n')
		fout.write('CELLSIZE 200\n')
		fout.write('NODATA_VALUE 0\n')
		info = ''
		for line in Map:
			info = ' '.join([str(x) for x in line]) + '\n' + info
		fout.write(info)
   
####################################################################################################################################################  
####################################################################################################################################################
####################################################################################################################################################  
####################################################################################################################################################
#
#   Module 3 : Running the main functions
#

# main functions:
classify_familiar_trial()

# main_animal_predict("紫貂")
main_poaching_predict(qgis_file_in)
prep_qgis(qgis_file_in, qgis_file_out)


