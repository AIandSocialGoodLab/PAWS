import matplotlib.pyplot as plt
import openpyxl as px
import numpy as np
import cPickle
import pdb
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='./data/', help='path to directory with data')
opt = parser.parse_args()

output = opt.data_dir+'/processed_forest_trial_new_deploy.pkl'
w = px.load_workbook(opt.data_dir+'/toydata.xlsx')
p = w.get_sheet_by_name('Sheet1')

row_num = 0
PositiveData = {}
NegativeData = {}
UnknownData = {}
deployNegative = {}
deployPositive = {}

Invalid_ID = []

ID_coordinate = {}
ID_isroad = {}
ID_elevation = {}
ID_slope = {}

fields = ['ID','distStream','elevation','slope','dist-boundary','dist-village','dist-patrol','dist-river','dist-marsh','dist-villagerode','dist-highway','dist-national','dist-provice','dist_patrol14','dist_patrol15','if_patrol16','dist_patrol17','dist_patrol18','if_poaching14','if_poaching15','if_poaching16','if_poaching17','if_poaching18']

for i, row in enumerate(p.rows):
    line=[]
    if i>0:
        for j, field in enumerate(fields):
            line.append(row[j].internal_value)

        patrol_length = np.sum(line[13:18])
        if patrol_length > 0:
            if line[19] > 0:
                PositiveData[300000+int(line[0])] = line[1:13] + [line[13]]
            else:
                NegativeData[300000+int(line[0])] = line[1:13] + [line[13]]
            if line[20] > 0:
                PositiveData[200000+int(line[0])] = line[1:13] + [line[14]]
            else:
                NegativeData[200000+int(line[0])] = line[1:13] + [line[14]]
            if line[21] > 0:
                PositiveData[100000+int(line[0])] = line[1:13] + [line[15]]
            else:
                NegativeData[100000+int(line[0])] = line[1:13] + [line[15]]
            if line[22] > 0:
                PositiveData[400000+int(line[0])] = line[1:13] + [line[16]]
            else:
                NegativeData[400000+int(line[0])] = line[1:13] + [line[16]]
        else:
            UnknownData[int(line[0])] = line[1:13] + [0]
                #pdb.set_trace()

cPickle.dump([PositiveData, NegativeData, UnknownData],
             open(output, 'w'))

print 'Labeled data number:', len(PositiveData) + len(NegativeData)
print 'Labeled positive data:', len(PositiveData)
print 'Labeled negative data:', len(NegativeData)
print 'Saved data to %s' % output
