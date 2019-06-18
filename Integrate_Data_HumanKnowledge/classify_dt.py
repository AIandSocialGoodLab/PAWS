import sys
from dataset import DataSet
#from SpecialDecisionTreeClassifier import *
import matplotlib.pyplot as plt
import pickle as cPickle
import numpy as np
from sklearn import tree
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import roc_auc_score
from scipy import optimize
import argparse


FoldNum = 4
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='./data/', help='path to directory with data')
opt = parser.parse_args()
PositiveData, NegativeData, UnknownData = cPickle.load(open(opt.data_dir+'/processed_forest_trial_new_deploy.pkl','rb'))
clusters = np.load(opt.data_dir+'/cluster40_scores.npy').item()
clusters_50 = np.load(opt.data_dir+'/cluster50_scores.npy').item()
clusters[0]=9
clusters_50[0] = 9
num_clusters=40
id_label = {}
cluster_ids = {}
cluster_ids50 = {}
for i in range(10):
    cluster_ids[i]=[]
    cluster_ids50[i]=[]
with open(opt.data_dir+'/result_predict_i'+str(num_clusters)+'.txt') as fin:
    fin.readline()
    for line in fin:
        line = line.strip().split()
        index = int(float(line[0]))
        label = int(float(line[1]))
        cluster_ids[clusters[label]].append(index)
with open(opt.data_dir+'/result_predict_i'+str(50)+'.txt') as fin:
    fin.readline()
    for line in fin:
        line = line.strip().split()
        index = int(float(line[0]))
        label = int(float(line[1]))
        cluster_ids50[clusters_50[label]].append(index)

Pdata = []
DPdatal = []
Ndata = []
DNdatal = []
Udata = []
PdataID = []
DPdataID = []
NdataID = []
DNdataID = []
UdataID = []
for i in range(500000):
    if i in PositiveData:
        Pdata.append(PositiveData[i])
        PdataID.append(i%100000)
    if i in NegativeData:
        Ndata.append(NegativeData[i])
        NdataID.append(i%100000)
    if i in UnknownData:
        Udata.append(UnknownData[i])
        UdataID.append(i%100000)

PositiveData = np.array(Pdata)
#PositiveData[:,12] = 0 
NegativeData = np.array(Ndata)
#NegativeData[:,12] = 0
UnknownData = np.array(Udata)
#UnknownData[:,12] = 0
maxvals = np.max(np.array(Udata+Pdata+Ndata),0,keepdims=True)[0:1]
maxvals[maxvals==0]=1.
PositiveData = PositiveData/maxvals
NegativeData = NegativeData/maxvals
UnknownData = UnknownData/maxvals
PositiveDataID = np.array(PdataID)
NegativeDataID = np.array(NdataID)
UnknownDataID = np.array(UdataID)
'''
#### Use during test time
ALLDATA = Udata +Pdata + Ndata
ALLID = UdataID + PdataID + NdataID
index = np.random.permutation(len(PositiveData))
PositiveData = PositiveData[index]
index = np.random.permutation(len(NegativeData))
NegativeData = NegativeData[index]
sample_size = NegativeData.shape[0]
indx = np.random.randint(UnknownData.shape[0], size=sample_size)
Udata = UnknownData[indx]
NotFam = NegativeData
#NotFam = np.concatenate((Udata, NegativeData), axis=0)
Fam = PositiveData
Fam = np.concatenate((PositiveData,PositiveData[PositiveData[:,2]>0.5]))
Fam = np.concatenate((Fam,PositiveData[PositiveData[:,3]>0.47]))
dataset = DataSet(positive=Fam, negative=NotFam, fold_num=FoldNum)
train_data, train_label = dataset.get_train_all_up(40)
clf = BaggingClassifier(tree.DecisionTreeClassifier(), n_estimators=1000, max_samples=0.1)
clf.fit(train_data, train_label)
ALLDATA = np.array(ALLDATA)
#ALLDATA[:,12]=0
print ALLDATA.shape
ALL_scores = clf.predict(ALLDATA)
ALL_value = clf.predict_proba(ALLDATA)[:,1]
unl_scores = clf.predict(UnknownData)
neg_scores = clf.predict(NegativeData)
pos_scores = clf.predict(PositiveData)
print np.sum(pos_scores)
print np.sum(neg_scores)
print np.sum(unl_scores)
print np.shape(ALL_value), np.max(ALL_value)
id_label = zip(ALLID, ALL_value)
for id in Invalid_ID:
    id_label.append((id, 0.0))
id_label = sorted(id_label, key=lambda x: x[0], reverse=False)


#with open('result_predict1.txt', 'w') as fout:
#    fout.write('ID\tLabel\n')
#    for idx, label in id_label:
#        fout.write(str(idx) + '\t' + str(label) + '\n')

'''
fold_pos_num = len(PositiveData) // int(FoldNum)
fold_neg_num = len(NegativeData) // int(FoldNum)
neg = NegativeData[:fold_neg_num]
NegativeData = NegativeData[fold_neg_num:]

###############################################################################

def unl_proba(UnknownData, maxvals):
  #mean_thresh = [0.09, 0.1, 0.2, 0.3, 0.5, 0.05, 0.37, 0.3]
  #index = [2, 3, 9, 9, 8, 5, 10, 11]
  mean_thresh = [0.3, 0.03, 0.03, 0.1, 0.1, 200/maxvals[0,10], 25/maxvals[0,11]]
  index = [2, 3, 5, 8, 9, 10, 11] 
  criterion = UnknownData[:,index[:2]]>mean_thresh[:2]
  criterion1 = UnknownData[:,index[5:]]<mean_thresh[5:]
  criterion = np.prod(criterion*0.8 + (1-criterion)*0.2,axis=1)
  criterion1 = np.prod(criterion1*0.8 + (1-criterion1)*0.2,axis=1)*criterion
  return criterion1
p = unl_proba(UnknownData,maxvals)
index = np.argsort(p,axis=None)

################################################################################

sample_size = NegativeData.shape[0]
indx = np.random.randint(UnknownData.shape[0], size=sample_size)
Udata = UnknownData[indx]
UdataNeg = []
'''
#### Augmenting the negative data using Unlabeled examples with low cluster scores
for i,id in enumerate(UnknownDataID.reshape(-1)):
    if (id in cluster_ids[1] or id in cluster_ids[0] or id in cluster_ids[2] or id in cluster_ids[3]) and (id in cluster_ids50[0] or id in cluster_ids50[1] or id in cluster_ids50[2]):# or id in cluster_ids50[3]):
        UdataNeg.append(UnknownData[i:i+1,:])
UdataNeg=np.concatenate(UdataNeg,0)
Udata = np.concatenate((UdataNeg, Udata),0)
'''
NotFam = np.concatenate((Udata, NegativeData), axis=0)
#NotFam = NegativeData
neg_label = np.array([0.] * len(neg))
Fam = PositiveData
dataset = DataSet(positive=Fam, negative=NotFam, fold_num=FoldNum)
auc_list = []
for iters in range(5):
    dataset = DataSet(positive=Fam, negative=NotFam, fold_num=FoldNum)
    for fold_id in range(FoldNum):
        dataset.update_negative(NotFam)
        y_prob_sum = 0
        for j in range(1):
            train_data, train_label, test_data_pos, test_label_pos = dataset.get_train_neg_traintest_pos_aug(cluster_ids, cluster_ids50, index, UnknownData, UnknownDataID, fold_id, 350)
            #train_data, train_label, test_data_pos, test_label_pos = dataset.get_train_neg_traintest_pos(fold_id, 300)   # Without Positive sampling
            test_data = np.concatenate((test_data_pos, neg), axis=0)
            #test_data[:,12]=0
            test_label = np.concatenate((test_label_pos, neg_label), axis=0)
            clf = BaggingClassifier(tree.DecisionTreeClassifier(criterion = "entropy"), n_estimators=1000, max_samples=0.1)
            clf.fit(train_data, train_label)
            unl_scores = clf.predict_proba(UnknownData)[:,1]
            y_prob = clf.predict_proba(test_data)
            y_prob_sum = y_prob
            y_pred = np.argmax(y_prob_sum,1)
            auc = roc_auc_score(y_true=test_label, y_score=y_pred)
            tp = np.sum(y_pred*test_label)
            fn = np.sum((1-y_pred)*test_label)
            denominator = np.sum(y_pred)/np.float(len(y_pred))
            lnl = np.square((tp/(tp+fn)))/denominator
            fp = np.sum(y_pred*(1-test_label))
            '''
            ### Dynamic Negative Sampling
            y_neg_scores = clf.predict_proba(train_data)
            #fp_train = np.sum((y_neg_scores>0.5)*(1-train_label))
            #sample_size = int((fp_train-200)/50) +5
            unl_score_mean = unl_scores
            indx = np.random.randint(UnknownData.shape[0], size=5)
            maxidx = np.argmax(unl_score_mean[indx])
            Udata = UnknownData[[indx[maxidx]]]
            for i in range(sample_size):
                indx = np.random.randint(UnknownData.shape[0], size =5)
                maxidx = np.argsort(unl_score_mean[indx])
                Udata = np.concatenate((Udata, UnknownData[[indx[maxidx[-1]]]]), axis=0)
            NotFam1 = np.concatenate((Udata, NegativeData), axis=0)
            dataset.update_negative(NotFam1)
            '''
            print("auc", np.mean(auc), "lnl", lnl,"Recall", tp/(tp+fn))
        auc_list.append([auc,lnl,tp/(tp+fn),tp/(tp+fp)])
    print('mean_auc', np.mean(auc_list,axis=0))

# Bagging UpSample 0.85726173772
