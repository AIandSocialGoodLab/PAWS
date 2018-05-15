
Implementation of "Exploiting Data and Human Knowledge for Predicting Wildlife Poaching". (COMPASS 2018)

### Dependencies
'''python
numpy==1.14.1
cPickle==1.71
scipy==1.0.0
sklearn==0.0
tensorflow==1.4
matplotlib==2.1.0
'''
### File Descriptions
classify_familiar_trial.py - This file has an implementation of the Bagging Ensemble of Decision Trees. The implementation also uses the domain knowledge (clusters and individual feature thresholds) provided by experts during training. 

classify_familiar_nn.py - This file implements a neural network ensemble replacing the decision tree in classify_familiar_trial.py

clustering.py - This file implements k-means algorithm to cluster the data points into various clusters. These clusters were given to the domain experts to evaluate the cluster threat probabilities.

dataset.py - This file contains a data structure to process and store the input data. It provides the train - test - validation splits in various forms for different kinds of training procedures. It also has functions to duplicate and balance the data.

make_data_forest_modified_deploy.py - This file is modification of make_data_forest_modified.py and adds a dictionary for the deployed data. 

### Run Code
To run any of the files in the repository, run the following command.
'''
python3 filename --data_dir directory
'''

