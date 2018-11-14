import numpy as np
import pdb
import time


class DataSet(object):
    def __init__(self, positive, negative, fold_num):
        self.positive = np.array(positive)
        self.negative = np.array(negative)
        self.fold_num = fold_num

        index = np.random.permutation(len(self.positive))
        self.positive = self.positive[index]
        index = np.random.permutation(len(self.negative))
        self.negative = self.negative[index]

        # Split k-fold
        self.data_folds = []
        self.label_folds = []
        fold_pos_num = int(len(self.positive) / int(fold_num))
        fold_neg_num = int(len(self.negative) / int(fold_num))

        for i in range(fold_num):
            if i == fold_num - 1:
                pos = self.positive[i * fold_pos_num:]
                neg = self.negative[i * fold_neg_num:]

                data = np.concatenate((pos, neg), axis=0)
                label = np.array([1.] * len(pos) + [0.] * len(neg))

                index = np.random.permutation(len(data))
                self.data_folds.append(data[index])
                self.label_folds.append(label[index])
            else:
                pos = self.positive[i * fold_pos_num:(i + 1) * fold_pos_num]
                neg = self.negative[i * fold_neg_num:(i + 1) * fold_neg_num]

                data = np.concatenate((pos, neg), axis=0)
                label = np.array([1.] * len(pos) + [0.] * len(neg))

                index = np.random.permutation(len(data))
                self.data_folds.append(data[index])
                self.label_folds.append(label[index])
    def update_negative(self, negative):
        self.negative = np.array(negative)
        index = np.random.permutation(len(self.negative))
        self.negative = self.negative[index]
        fold_num = self.fold_num
        self.data_folds = []
        self.label_folds = []
        fold_pos_num = int(len(self.positive) / int(fold_num))
        fold_neg_num = int(len(self.negative) / int(fold_num))
        for i in range(fold_num):
            if i == fold_num - 1:
                pos = self.positive[i * fold_pos_num:]
                neg = self.negative[i * fold_neg_num:]

                data = np.concatenate((pos, neg), axis=0)
                label = np.array([1.] * len(pos) + [0.] * len(neg))

                index = np.random.permutation(len(data))
                self.data_folds.append(data[index])
                self.label_folds.append(label[index])
            else:
                pos = self.positive[i * fold_pos_num:(i + 1) * fold_pos_num]
                neg = self.negative[i * fold_neg_num:(i + 1) * fold_neg_num]

                data = np.concatenate((pos, neg), axis=0)
                label = np.array([1.] * len(pos) + [0.] * len(neg))

                index = np.random.permutation(len(data))
                self.data_folds.append(data[index])
                self.label_folds.append(label[index])

    def get_train_test(self, fold_id):
        data_folds_copy = list(self.data_folds)
        label_folds_copy = list(self.label_folds)

        test_data = data_folds_copy.pop(fold_id)
        test_label = label_folds_copy.pop(fold_id)

        train_data = np.concatenate(data_folds_copy, axis=0)
        train_label = np.concatenate(label_folds_copy, axis=0)
        return train_data, train_label, test_data, test_label

    def get_train_test_upsample(self, fold_id, num):
        data_folds_copy = list(self.data_folds)
        label_folds_copy = list(self.label_folds)

        test_data = data_folds_copy.pop(fold_id)
        test_label = label_folds_copy.pop(fold_id)

        train_data = np.concatenate(data_folds_copy, axis=0)
        train_label = np.concatenate(label_folds_copy, axis=0)

        train_data_up = []
        train_label_up = []
        for data, label in zip(train_data, train_label):
            if label:
                train_data_up += [data] * num
                train_label_up += [label] * num
            else:
                train_data_up.append(data)
                train_label_up.append(label)

        train_data_up = np.array(train_data_up)
        train_label_up = np.array(train_label_up)
        index = np.random.permutation(len(train_label_up))
        train_data_up = train_data_up[index]
        train_label_up = train_label_up[index]

        return train_data_up, train_label_up, test_data, test_label

    def get_train_all(self):
        data_folds_copy = list(self.data_folds)
        label_folds_copy = list(self.label_folds)

        train_data = np.concatenate(data_folds_copy, axis=0)
        train_label = np.concatenate(label_folds_copy, axis=0)
        return train_data, train_label

    def get_train_all_up(self, num):
        data_folds_copy = list(self.data_folds)
        label_folds_copy = list(self.label_folds)

        train_data = np.concatenate(data_folds_copy, axis=0)
        train_label = np.concatenate(label_folds_copy, axis=0)

        train_data_up = []
        train_label_up = []
        for data, label in zip(train_data, train_label):
            if label:
                train_data_up += [data] * num
                train_label_up += [label] * num
            else:
                train_data_up.append(data)
                train_label_up.append(label)

        train_data_up = np.array(train_data_up)
        train_label_up = np.array(train_label_up)
        index = np.random.permutation(len(train_label_up))
        train_data_up = train_data_up[index]
        train_label_up = train_label_up[index]

        return train_data_up, train_label_up
    def get_train_neg_traintest_pos(self,fold_id, num):
        #start = time.time()
        data_folds_copy = list(self.data_folds)
        label_folds_copy = list(self.label_folds)

        test_data1 = data_folds_copy.pop(fold_id)
        test_label1 = label_folds_copy.pop(fold_id)

        train_data = np.concatenate(data_folds_copy, axis=0)
        train_label = np.concatenate(label_folds_copy, axis=0)

        train_data_up = []
        train_label_up = []
        test_data = []
        test_label = []
        for data, label in zip(train_data, train_label):
            if label:
                train_data_up += [data]*num
                train_label_up += [label]*num
            else:
                train_data_up.append(data)
                train_label_up.append(label)
        for data, label in zip(test_data1,test_label1):
            if label:
                test_data.append(data)
                test_label.append(label)
            else:
                train_data_up.append(data)
                train_label_up.append(label)

        train_data_up = np.array(train_data_up)
        train_label_up = np.array(train_label_up)
        index = np.random.permutation(len(train_label_up))
        train_data_up = train_data_up[index]
        train_label_up = train_label_up[index]
        #print("duration" , time.time() - start)
        return train_data_up, train_label_up, test_data, test_label

    def get_train_neg_traintest_pos_smote(self,fold_id, num):
        #start = time.time()
        data_folds_copy = list(self.data_folds)
        label_folds_copy = list(self.label_folds)

        test_data1 = data_folds_copy.pop(fold_id)
        test_label1 = label_folds_copy.pop(fold_id)

        train_data = np.concatenate(data_folds_copy, axis=0)
        train_label = np.concatenate(label_folds_copy, axis=0)

        train_data_up = []
        train_label_up = []
        test_data = []
        test_label = []
        train_data_pos = []
        train_label_pos = []
        for data, label in zip(train_data, train_label):
            if label:
                train_data_pos += [data]
                train_label_pos += [label]
            else:
                train_data_up.append(data)
                train_label_up.append(label)
        for data, label in zip(test_data1,test_label1):
            if label:
                test_data.append(data)
                test_label.append(label)
            else:
                train_data_up.append(data)
                train_label_up.append(label)

        train_data_pos = np.array(train_data_pos)
        train_label_pos = np.array(train_label_pos)
        #pdb.set_trace()
        idx_sort = np.argsort(np.sum(np.square(np.expand_dims(train_data_pos,2)-np.tile(train_data_pos,(train_data_pos.shape[0],1)).reshape(train_data_pos.shape+(train_data_pos.shape[0],))),axis=1),axis=1)
        for j, (data, label) in enumerate(zip(train_data_pos, train_label_pos)):
            for i in range(num):
                a = np.random.uniform(0,1)
                idx = np.random.randint(len(train_data_pos))
                train_data_up += [data*a + (1-a)*train_data_pos[idx_sort[j,idx]]]
                train_label_up += [label]

        train_data_up = np.array(train_data_up)
        train_label_up = np.array(train_label_up)
        index = np.random.permutation(len(train_label_up))
        train_data_up = train_data_up[index]
        train_label_up = train_label_up[index]
        #print("smote_duration" , time.time() - start)
        return train_data_up, train_label_up, test_data, test_label
    def get_train_all_up_aug(self,UdataPos, num):
        data_folds_copy = list(self.data_folds)
        label_folds_copy = list(self.label_folds)

        train_data = np.concatenate(data_folds_copy, axis=0)
        train_label = np.concatenate(label_folds_copy, axis=0)
        # UdataPos = []
        # UdataPos1 = []
        '''
        for i,id in enumerate(UnknownDataID.reshape(-1)):
            if (id in cluster_ids[8] or id in cluster_ids[7]) and (id in cluster_ids50[7] or id in cluster_ids50[6]):
                UdataPos.append(UnknownData[i:i+1,:])
        #        UdataPos.append(UnknownData[i:i+1,:])
        '''
        #UdataPos.append(UnknownData[index[:1000]])
        # for i,id in enumerate(UnknownDataID.reshape(-1)):
        #     if (id in cluster_ids[8] or id in cluster_ids[7] or id in cluster_ids[6]) and (id in cluster_ids50[7] or id in cluster_ids50[6]):
        #         UdataPos.append(UnknownData[i:i+1,:])
        # #        UdataPos.append(UnknownData[i:i+1,:])
        # UdataPos=np.concatenate(UdataPos,0)
        # print(len(UdataPos))

        train_data_up = []
        train_label_up = []
        for data, label in zip(train_data, train_label):
            if label:
                train_data_up += [data] * num
                train_label_up += [label] * num
                label1 = label
            else:
                train_data_up.append(data)
                train_label_up.append(label)
        for data in UdataPos:
            train_data_up +=[data]
            train_label_up += [label1]

        train_data_up = np.array(train_data_up)
        train_label_up = np.array(train_label_up)
        index = np.random.permutation(len(train_label_up))
        train_data_up = train_data_up[index]
        train_label_up = train_label_up[index]

        return train_data_up, train_label_up
 
    def get_train_neg_traintest_pos_aug(self, cluster_ids, cluster_ids50, index, UnknownData, UnknownDataID, fold_id, num):
        data_folds_copy = list(self.data_folds)
        label_folds_copy = list(self.label_folds)

        test_data1 = data_folds_copy.pop(fold_id)
        test_label1 = label_folds_copy.pop(fold_id)

        train_data = np.concatenate(data_folds_copy, axis=0)
        train_label = np.concatenate(label_folds_copy, axis=0)

        UdataPos = []
        '''
        for i,id in enumerate(UnknownDataID.reshape(-1)):
            if (id in cluster_ids[8] or id in cluster_ids[7]) and (id in cluster_ids50[7] or id in cluster_ids50[6]):
                UdataPos.append(UnknownData[i:i+1,:])
        #        UdataPos.append(UnknownData[i:i+1,:])
        '''
        #UdataPos.append(UnknownData[index[:1000]])
        for i,id in enumerate(UnknownDataID.reshape(-1)):
            if (id in cluster_ids[8] or id in cluster_ids[7] or id in cluster_ids[6]) and (id in cluster_ids50[7] or id in cluster_ids50[6]):
                UdataPos.append(UnknownData[i:i+1,:])
        #        UdataPos.append(UnknownData[i:i+1,:])
        UdataPos=np.concatenate(UdataPos,0)
        print(UdataPos.shape)
        train_data_up = []
        train_label_up = []
        test_data = []
        test_label = []
        for data, label in zip(train_data, train_label):
            if label:
                train_data_up += [data] * num
                train_label_up += [label] * num
                label1 = label
            else:
                train_data_up.append(data)
                train_label_up.append(label)
        for data, label in zip(test_data1,test_label1):
            if label:
                test_data.append(data)
                test_label.append(label)
            else:
                train_data_up.append(data)
                train_label_up.append(label)
        for data in UdataPos:
            train_data_up +=[data]
            train_label_up += [label1]
        train_data_up = np.array(train_data_up)
        train_label_up = np.array(train_label_up)
        index = np.random.permutation(len(train_label_up))
        train_data_up = train_data_up[index]
        train_label_up = train_label_up[index]
        return train_data_up, train_label_up, test_data, test_label

