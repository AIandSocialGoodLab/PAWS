from dataset import DataSet
import matplotlib.pyplot as plt
import cPickle
import numpy as np
from sklearn import tree
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import roc_auc_score
from scipy.cluster.vq import whiten
from scipy.cluster.vq import kmeans
import openpyxl as px
import argparse

FoldNum = 4
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='./data/', help='path to directory with data')
opt = parser.parse_args()

PositiveData, NegativeData, UnknownData, Invalid_ID, ID_coordinate, ID_isroad, ID_elevation, ID_slope= cPickle.load(open(opt.data_dir+'/processed_forest_trial_new.pkl'))

Pdata = []
Ndata = []
Udata = []
PdataID = []
NdataID = []
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
ALLDATA = Pdata + Ndata + Udata
ALLID = PdataID + NdataID + UdataID
ALLDATA = np.array(ALLDATA)
num_clusters = 50
wh_data = whiten(ALLDATA[:,:12])
centroid,distortion = kmeans(wh_data,num_clusters)
label = np.zeros(wh_data.shape[0])
id_label = []
for i in range(wh_data.shape[0]):
    data1 = np.array([wh_data[i]])
    label[i] = np.argmin(np.sum(np.square(centroid-data1),1))
#for i in range(num_clusters):
labeli = label+1#(label==i) +1-1
#print(np.sum(labeli))
labeli=zip(ALLID, labeli)
invalid=zip(Invalid_ID,[0.0]*len(Invalid_ID))
id_label=sorted(np.concatenate([labeli,invalid],0), key=lambda x:x[0], reverse=False)
with open(opt.data_dir+'/result_predict_i'+str(num_clusters)+'.txt', 'w')as fout:
    fout.write('ID\tLabel\n')
    for idx, labels in id_label:
        fout.write(str(idx) + '\t' + str(labels) + '\n')
wb = px.Workbook()
for j in range(num_clusters):
    id = label==j
    print(np.sum(id))
    data = ALLDATA[id,:12]
    print(data.shape)
    ws=wb.create_sheet('Cluster'+str(j+1))
    ws.append(['dist-Stream','dist-boundary','dist-village','dist-patrol','dist-riverJ2','dist-marsh','dist-villagerode','dist-highway','dist-national','dist-provice','elevation','slope'])
    for i in range(np.sum(id)):
        ws.append(list(data[i]))
wb.save(opt.data_dir+'/clustered_regions_'+str(num_clusters)+'.xlsx')
        
