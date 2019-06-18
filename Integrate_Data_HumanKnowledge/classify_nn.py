import tensorflow as tf
from dataset import DataSet
import cPickle
import numpy as np
from sklearn import tree
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import roc_auc_score
import argparse


FoldNum = 4
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='./data/', help='path to directory with data')
opt = parser.parse_args()

learning_rate = 0.001
training_epochs = 10
batch_size = 64
display_step = 1
ensemble_num =100
ensembles = 10
# Network Parameters
n_hidden_1 = 8 # 1st layer number of neurons
n_hidden_2 = 4 # 2nd layer number of neurons
n_input = 24 # MNIST data input (img shape: 28*28)
n_classes = 1 # MNIST total classes (0-9 digits)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

# tf Graph input
X = tf.placeholder("float", [None, n_input])
Y = tf.placeholder("float", [None, n_classes])
keep_prob = tf.placeholder("float")
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}
def multilayer_perceptron(x,scope):
    with tf.variable_scope(scope):
        # Hidden fully connected layer with 256 neurons
        layer_1 = tf.nn.relu(tf.add(tf.matmul(x, weights['h1']), biases['b1']))
        layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])) # Output fully connected layer with a neuron for each class
        out_layer = tf.add(tf.matmul(layer_2, weights['out']), biases['out'])
        l2_error = 0.1*tf.reduce_mean(tf.reduce_mean(tf.square(weights['h1']))) \
                   + 0.1*tf.reduce_mean(tf.reduce_mean(tf.square(weights['h2'])))
    return [out_layer, l2_error]
sig = []
logits = []
loss_op = []
train_op = []
for i in range(ensemble_num):
    logit, l2_error = multilayer_perceptron(X,str(i))
    logits.append(logit)
    sig.append(tf.nn.sigmoid(logit))

    # Define loss and optimizer
    loss_op.append(tf.reduce_mean(tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = logit, labels=Y)))+l2_error)
    train_op.append(optimizer.minimize(loss_op[-1]))

# Initializing the variables
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)


PositiveData, NegativeData, UnknownData = cPickle.load(open(opt.data_dir+'/processed_forest_trial_new_deploy.pkl','rb'))

Pdata = []
Ndata = []
Udata = []
for i in range(5000000):
    if i in PositiveData:
        Pdata.append(PositiveData[i])
    if i in NegativeData:
        Ndata.append(NegativeData[i])
    if i in UnknownData:
        Udata.append(UnknownData[i])

PositiveData = np.array(Pdata)
NegativeData = np.array(Ndata)
UnknownData = np.array(Udata)
PositiveData[:,12] = 0
NegativeData[:,12] = 0
UnknownData[:,12] = 0
maxvals = np.max(np.array(Udata+Pdata+Ndata),0,keepdims=True)[0:1]
maxvals[maxvals==0]=1.
PositiveData = PositiveData/maxvals
NegativeData = NegativeData/maxvals
UnknownData = UnknownData/maxvals
#print(UnknownData.shape,NegativeData.shape)
index = np.random.permutation(len(PositiveData))
PositiveData = PositiveData[index]
index = np.random.permutation(len(NegativeData))
NegativeData = NegativeData[index]
fold_pos_num = int(len(PositiveData) / int(FoldNum))
fold_neg_num = int(len(NegativeData) / int(FoldNum))
pos = PositiveData[:fold_pos_num]
neg = NegativeData[:fold_neg_num]
print(fold_pos_num, fold_neg_num)
test_data = np.concatenate((pos, neg), axis=0)
test_label = np.array([1.] * len(pos) + [0.] * len(neg))

PositiveData = PositiveData[fold_pos_num:]
NegativeData = NegativeData[fold_neg_num:]

sample_size = NegativeData.shape[0]
indx = np.random.randint(UnknownData.shape[0], size=sample_size)
Udata = UnknownData[indx]
#print(Udata.shape, NegativeData.shape)
NotFam = np.concatenate((Udata, NegativeData), axis=0)
Fam = PositiveData
#NotFam = np.array(Ndata)
y_prob_sum = np.zeros((test_label.shape[0],2))
dataset = DataSet(positive=Fam, negative=NotFam, fold_num=FoldNum)
unl_scores = np.zeros((UnknownData.shape[0],FoldNum))
auc_list = []
#saver = tf.train.Saver()
y_prob_acc=np.zeros((test_label.shape[0],1))
test_label = np.reshape(test_label, (test_label.shape[0],1))
with tf.Session() as sess:
    max_auc=0.
    for e_id in range(ensemble_num):
        dataset = DataSet(positive=Fam, negative=NotFam, fold_num=FoldNum)
        sess.run(init)
        for j in range(1):
            # Training cycle
            epoch = 0
            while epoch<training_epochs:# or auc<max_auc-0.02:
                train_data, train_label = dataset.get_train_all_up(160)
                avg_cost = 0.
                total_batch = int(train_data.shape[0]/batch_size)
                # Loop over all batches
                for i in range(total_batch):
                    batch_x = train_data[i*batch_size:(i+1)*batch_size,:]
                    batch_y = train_label[i*batch_size:(i+1)*batch_size]
                    batch_y = batch_y.reshape(batch_y.shape[0],1)
                    # Run optimization op (backprop) and cost op (to get loss value)
                    _, c = sess.run([train_op[e_id], loss_op[e_id]], feed_dict={X: batch_x,
                                                            Y: batch_y, keep_prob:0.5})

                y_prob = sig[e_id].eval({X: test_data, keep_prob:1.0})
                auc = roc_auc_score(y_true=test_label, y_score=(y_prob>0.5))
                if auc>=max_auc:
                    max_auc=auc
                epoch +=1
                tp = np.sum((y_prob>0.5)*test_label)
                denominator = np.float(np.sum(y_prob>0.5))/len(y_prob)
                fn = np.float(np.sum((y_prob<=0.5)*test_label))
                lnl =np.square((tp/(tp+fn)))/denominator
                fp = np.sum((y_prob>0.5)*(1-test_label))
            print('model: ', e_id, ' iteration:', j, ' auc: ', auc, ' l&l: ', tp/(tp+fn),tp/(tp+fp), np.sum(y_prob>=0.5), denominator, fn, test_label.shape)

            unl_scores = sig[e_id].eval({X: UnknownData, keep_prob:1.0})
            indx = np.random.randint(UnknownData.shape[0], size=10)
            maxidx = np.argmax(unl_scores[indx])
            Udata = UnknownData[[indx[maxidx]]]
            for i in range(sample_size):
                indx = np.random.randint(UnknownData.shape[0], size =10)
                maxidx = np.argsort(unl_scores[indx])
                Udata = np.concatenate((Udata, UnknownData[[indx[maxidx[9]]]]), axis=0)
            NotFam1 = np.concatenate((Udata, NegativeData), axis=0)
            dataset = DataSet(positive=Fam, negative=NotFam1, fold_num=FoldNum)

            # Display logs per epoch step
        y_prob_acc += y_prob
    #save_path = saver.save(sess, "./model.ckpt")
    #print("Model saved in file: %s" % save_path)
    # Test model
    y_prob = y_prob_acc/ensemble_num
    auc = roc_auc_score(y_true=test_label, y_score=(y_prob>0.5))
    tp = np.sum((y_prob>0.5)*test_label)
    denominator = np.float(np.sum(y_prob>0.5))/len(y_prob)
    fn = np.float(np.sum((y_prob<=0.5)*test_label))
    lnl =np.square((tp/(tp+fn)))/denominator
    fp = np.sum((y_prob>0.5)*(1-test_label))
    print("Accuracy:", auc, "ll", lnl, "Recall", tp/(tp+fn), "Precision",tp/(tp+fp))

# Bagging UpSample 0.85726173772
