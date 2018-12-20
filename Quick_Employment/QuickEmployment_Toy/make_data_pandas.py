import numpy as np
import pandas as pd
import sys
#from sklearn import tree
#from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import roc_auc_score
from dataset import DataSet
import xgboost as xgb

####################################################################################################################################################  
####################################################################################################################################################
####################################################################################################################################################  
####################################################################################################################################################
#
#         General Overview
#	Module -1: Processing output files from automate_data.py
#   Module 0 : Specifying feature names and output file names
#   Module 1 : Preprocessing
#   Module 2 : Defining the main functions
#   Module 3 : Running the main functions
#
# 	We assume that the user has one conservation site with labeled data (conservation site 1) concerning where poaching
#	and past patrol efforts have occurred, and use this knowledge to predict where future illegal attempts will be made in
#	both conservation site 1 and the unlabeled area (conservation site 2), assuming their feature spaces are the same.
#	We employ dynamic negative sampling of the unknown data, oversampling of the positive examples, and xgboost.
#

####################################################################################################################################################  
####################################################################################################################################################
####################################################################################################################################################  
####################################################################################################################################################

#
#   Module -1 : Processing output files from automate_data.py
#
#	
#

files = ['/Users/tianyug/Desktop/QuickEmployment_Toy/csv_output/X.csv', '/Users/tianyug/Desktop/QuickEmployment_Toy/csv_output/Y.csv', '/Users/tianyug/Desktop/QuickEmployment_Toy/csv_output/is-toy_patrol.csv', '/Users/tianyug/Desktop/QuickEmployment_Toy/csv_output/is-toy_poaching.csv', '/Users/tianyug/Desktop/QuickEmployment_Toy/csv_output/is-toy_road.csv', '/Users/tianyug/Desktop/QuickEmployment_Toy/csv_output/dist-toy_patrol.csv', '/Users/tianyug/Desktop/QuickEmployment_Toy/csv_output/dist-toy_poaching.csv', '/Users/tianyug/Desktop/QuickEmployment_Toy/csv_output/dist-toy_road.csv', '/Users/tianyug/Desktop/QuickEmployment_Toy/csv_output/toy_altitude.csv']

column_names = ['X', 'Y', 'is-toy_patrol', 'is-toy_poaching', 'is-toy_road', 'dist-toy_patrol', 'dist-toy_poaching', 'dist-toy_road', 'toy_altitude']

raw_df_list = []

# read files into dataframe
for f in files:
  print (f)
  raw_df_list.append(pd.read_csv(f))

# get the DN as a dataframe
DN_df = raw_df_list[0][['DN']].sort_values(by=['DN'])
DN_df.reset_index(inplace=True)

# rename columns and sort based on DN
select_df_list = []
for i in range(0,len(raw_df_list)):
  # rename columns
  col_names = ['DN',column_names[i]]
  raw_df_list[i].columns = col_names

  # sort by DN
  cur_sorted_df = raw_df_list[i].sort_values(by=['DN'])
  cur_sorted_df.reset_index(inplace=True)

  # select revelant columns
  cur_select_df = cur_sorted_df[[column_names[i]]]

  # normalize the selected columns
  cur_normalized_df = (cur_select_df - cur_select_df.min())/(cur_select_df.max()-cur_select_df.min())
  cur_normalized_df.columns = ["normal-"+column_names[i]]

  select_df_list.append(cur_select_df)
  if column_names[i][0:3] != 'is-': 
    select_df_list.append(cur_normalized_df)


# concatenate columns
select_df_list = [DN_df] + select_df_list
comb_DN_ABC = pd.concat(select_df_list, axis=1)
comb_DN_ABC.sort_values(by=["DN"],inplace=True)
comb_DN_ABC.drop(['index'], axis=1)
comb_DN_ABC.to_csv("final.csv") 

# sys.exit()

####################################################################################################################################################  
####################################################################################################################################################
####################################################################################################################################################  
####################################################################################################################################################




#
#   Module 0 : Specifying paths and file names
#
#	The following are placeholders/examples
#

# name of excel sheet containing all the features and labels for conservation 1 and 2
fn1 = "final.csv"
fn2 = "final.csv"

# name of text file output for probabilistic predictions of 
# each grid cell in conservations 1 and 2 
qgis_file_in1 = "predictions1.txt"
qgis_file_in2 = "predictions2.txt"
# raster file of probabilistic predictions
qgis_file_out1 = "predictions_heatmap1.asc"
qgis_file_out2 = "predictions_heatmap2.asc"
# specify which features to use from final.csv feature spreadsheet
selected_features = ["is-toy_road",
					 "normal-dist-toy_road",
					 "normal-toy_altitude"
					]
# specify which feature symbolizes where patrolling occurs
patrol = 'is-toy_patrol'
# specify which feature symbolizes where poaching occurs
poaching = 'is-toy_poaching'

# represents the coordinates of the left bottom corner for 
# conservation site 1 (longitude and latitude if working with WGS84)
xcorner1 = 127.76402335
ycorner1 = 43.5257568717
# represents the coordinates of the left bottom corner for 
# conservation site 2 (longitude and latitude if working with WGS84)
xcorner2 = 127.76402335
ycorner2 = 43.5257568717

# define the grid sizes (discretization levels) for each conservations ite
# which should match from the automate_data.py script
gridDim1 = 0.01
gridDim2 = 0.01


####################################################################################################################################################  
####################################################################################################################################################
####################################################################################################################################################  
####################################################################################################################################################
#
#   Module 1 : Preprocessing
#

df_alldata = pd.read_csv(fn1)

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


######################################################################################################################################################

df_alldata2 = pd.read_csv(fn2)

# select data without nan
df_validdata2 = df_alldata2.dropna()

# select data with nan 
df_invaliddata2 = df_alldata2[df_alldata2.isnull().any(axis=1)]

df_slct_valid = df_alldata2[selected_features]

NewAllData = df_slct_valid.values


######################################################################################################################################################

# number of folds specified for classify_familiar_trial
FoldNum = 4

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

NotFam = np.concatenate((Udata, NegativeData), axis=0)
neg_label = np.array([0.] * len(neg))
Fam = PositiveData

dataset = DataSet(positive=Fam, negative=NotFam, fold_num=FoldNum)

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
			for j in range(1):
				train_data, train_label, test_data_pos, test_label_pos = dataset.get_train_neg_traintest_pos_smote(fold_id, 300)
				test_data = np.concatenate((test_data_pos, neg), axis=0)
				
				test_label = np.concatenate((test_label_pos, neg_label), axis=0)

				##################### xgboost ##################################
				D_train = xgb.DMatrix(train_data, label = train_label)
				D_test = xgb.DMatrix(test_data, label = test_label)

				param = {'max_depth': 10, 'eta': 0.1, 'silent': 1, 'objective': 'binary:logistic'}
				num_round = 1000

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

				# print("test:", np.mean(test_auc), test_lnl, test_recall, \
				# 				test_per, test_f1score)
				# print("train:", np.mean(train_auc), train_lnl, train_recall, \
				# 				train_per, train_f1score)

			test_auc_list.append([test_auc, test_lnl, test_recall, test_per, test_f1score])
			train_auc_list.append([train_auc, train_lnl, train_recall, train_per, train_f1score])

		# print('mean for test: ', np.mean(test_auc_list,axis=0))
		# print('mean for train: ', np.mean(train_auc_list,axis=0))
		cur_auc_list = np.mean(test_auc_list,axis=0)
		total_list = total_list + cur_auc_list

	total_list = total_list / numIter
	print('mean over all iteration (test)', total_list)
	print("average auc_score: %f" % total_list[0])
	print("average lnl score: %f" % total_list[1])
	print("average recall: %f" % total_list[2])
	print("average precision: %f" % total_list[3])
	print("average f1score: %f" % total_list[4])	

# Generate the actual predictions by training on all the data
# qgis_file_in is the output of this function, which contains the probabilistic
# predictions as a text file
def main_poaching_predict(qgis_file_in1, qgis_file_in2):
	print("start main_poaching_predict")

	PositiveDataID = df_allpositive["DN"].values
	NegativeDataID = df_allnegative["DN"].values
	UnknownDataID = df_unknowndata["DN"].values
	NewAllDataID = df_validdata2["DN"].values

	ALLID = list(PositiveDataID) + list(NegativeDataID) + list(UnknownDataID)
	ALLDATA = list(df_slct_positive.values) + \
			  list(df_slct_negative.values) + \
			  list(df_slct_unlabeled.values)

	NEWALLID = list(NewAllDataID)
	NEWALLDATA = list(df_slct_valid.values)

	
	train_data, train_label = dataset.get_train_all_up(100)

	param = {'max_depth': 10, 'eta': 0.1, 'silent': 1, 'objective': 'binary:logistic'}
	num_round = 1000

	D_train = xgb.DMatrix(train_data, label = train_label)

	bst = xgb.train(param, D_train, num_round)

	########################################################################################################################

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

	Invalid_ID = df_invaliddata["DN"].values
	for id in Invalid_ID:
		id_label.append((id, 0.0))

	id_label = sorted(id_label, key=lambda x: x[0], reverse=False)
	# print (id_label)
	with open(qgis_file_in1, 'w')as fout:
		fout.write('ID\tLabel\n')
		for idx, label in id_label:
			temp_str = str(idx) + '\t' + str(label) + '\n'
			fout.write(temp_str)
	# metrics reporting
	###########################################################
	report_data = np.array(list(df_slct_positive.values) + list(df_slct_negative.values))

	report_label = np.array((df_slct_positive.shape[0]) * [1] + (df_slct_negative.shape[0]) * [0])

	D_report_data = xgb.DMatrix(report_data)

	report_value = bst.predict(D_report_data)

	report_scores = np.zeros(len(report_value))
	for i in range(0, len(report_value)):
		if (report_value[i] > 0.5):
			report_scores[i] = 1.0
		else:
			report_scores[i] = 0.0

	auc = roc_auc_score(y_true=report_label, y_score=report_value)

	tp = np.sum(report_scores * report_label)
	fn = np.sum((1-report_scores) * report_label)

	denominator = np.sum(report_scores)/np.float(len(report_scores))
	lnl = np.square((tp/(tp+fn)))/denominator
	fp = np.sum(report_scores*(1-report_label))
	f1score = 2*(tp/(tp+fp))*(tp/(tp+fn))/(tp/(tp+fp)+tp/(tp+fn))

	print("lnl:")
	print(lnl)

	print("recall:")
	print(tp/(tp+fn))

	print("precision")
	print(tp/(tp+fp))

	print("f1score:")
	print(f1score)
	

	########################################################################################################################

	NEWALLDATA = np.array(NEWALLDATA)

	D_NEWALLDATA = xgb.DMatrix(NEWALLDATA)

	# prediction results
	ALL_newvalue = bst.predict(D_NEWALLDATA)
	ALL_newscores = np.zeros(len(ALL_newvalue))

	for i in range(0, len(ALL_newvalue)):
		if (ALL_newvalue[i] > 0.5):
			ALL_newscores[i] = 1.0
		else:
			ALL_newscores[i] = 0.0

	newid_label = zip(NEWALLID, ALL_newvalue)
	newid_label = list(newid_label)

	Invalid_ID = df_invaliddata2["DN"].values
	for id in Invalid_ID:
		newid_label.append((id, 0.0))

	newid_label = sorted(newid_label, key=lambda x: x[0], reverse=False)
	# print (id_label)
	with open(qgis_file_in2, 'w')as fout:
		fout.write('ID\tLabel\n')
		for idx, label in newid_label:
			temp_str = str(idx) + '\t' + str(label) + '\n'
			fout.write(temp_str)\

# translates output from main_poaching_predict into an ASC file
def prep_qgis(qgis_file_in, qgis_file_out, cellsize, Xcorner, Ycorner, data):
	print("start prep_qgis")
	l_id = data["DN"].values
	l_X = data["X"].values
	l_Y = data["Y"].values

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
	min_x = int(min(x_set) / cellsize)
	min_y = int(min(y_set) / cellsize)
	max_x = int(max(x_set) / cellsize)
	max_y = int(max(y_set) / cellsize)

	#print("min_x: ", min_x, " max_x: ", max_x, " min_y: ", min_y, " max_y: ", max_y)

	dim = 1 + int((max(x_set) - min(x_set)) / cellsize)
	Map_index = np.zeros([dim, dim])
	Map = np.zeros([dim, dim])
	
	# Load target list
	id_label = {}

	with open(qgis_file_in) as fin:
		fin.readline()
		for line in fin:
			line = line.strip().split()
			#print(line[0])
			index = int(float(line[0]))
			label = float(line[1])
			id_label[index] = label

	valid = 0
	count = 0
	coincides = 0
	for index in ID_coordinate:
		id_x = int((ID_coordinate[index][0] - min(x_set)) / cellsize)
		id_y = int((ID_coordinate[index][1] - min(y_set)) / cellsize)
		try:
			valid += 1
			if Map[id_y, id_x] > 1E-20:
				coincides += 1
				Map[id_y, id_x + 1] = id_label[index]
			else: 
				Map[id_y, id_x] = id_label[index]	
		except:
			count += 1
			# print index
			Map[id_y, id_x] = 0.0

	print("number of key error: %d" % count, "  number of valid: ", valid, "  number of coincides: ", coincides)
	
	with open(qgis_file_out, 'w') as fout:
		fout.write('NCOLS ' + str(dim) + '\n')
		fout.write('NROWS ' + str(dim) + '\n')
		fout.write('XLLCORNER ' + str(Xcorner) + '\n')
		fout.write('YLLCORNER ' + str(Ycorner) + '\n')
		fout.write('CELLSIZE ' + str(cellsize) + '\n')
		fout.write('NODATA_VALUE 0\n')
		info = ''
		for line in Map:
			info = ' '.join([str(x) for x in line]) + '\n' + info
		fout.write(info)
		
	print("done preparing")

####################################################################################################################################################  
####################################################################################################################################################
####################################################################################################################################################  
####################################################################################################################################################
#
#   Module 3 : Running the main functions
#

# classify_familiar_trial()
main_poaching_predict(qgis_file_in1, qgis_file_in2)
prep_qgis(qgis_file_in1, qgis_file_out1, gridDim1, xcorner1, ycorner1, df_alldata)
# comment this next line if we are not  testing on an unlabled conservation site
# prep_qgis(qgis_file_in2, qgis_file_out2, gridDim2, xcorner2, ycorner2, df_alldata2)


