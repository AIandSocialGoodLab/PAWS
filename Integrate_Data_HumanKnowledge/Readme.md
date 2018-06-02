
Implementation of "Exploiting Data and Human Knowledge for Predicting Wildlife Poaching". (COMPASS 2018)

### Dependencies
```python
numpy==1.14.1
cPickle==1.71
scipy==1.0.0
sklearn==0.0
tensorflow==1.4
matplotlib==2.1.0
```
### File Descriptions
classify_dt.py - This file has an implementation of the Bagging Ensemble of Decision Trees. The implementation also uses the domain knowledge (clusters and individual feature thresholds) provided by experts during training. 

classify_nn.py - This file implements a neural network ensemble replacing the decision tree in classify_dt.py

clustering.py - This file implements k-means algorithm to cluster the data points into various clusters. These clusters were given to the domain experts to evaluate the cluster threat probabilities.

dataset.py - This file contains a data structure to process and store the input data. It provides the train - test - validation splits in various forms for different kinds of training procedures. It also has functions to duplicate and balance the data.

make_data.py - This file preprocesses the excel file which contains the data and stores it as a set of dictionareis in a .pkl file. 

### Run Code
classify_dt.py, classify_nn.py, clustering.py and make_data.py can be run using the following command.
```
python3 filename --data_dir directory
```
where the `filename` should be replaced with the name of the corresponding file to be run and the `directory` should be replaced with the directory path where the data is stored.

For illustration purposes some randomly generated data has been provided in the `toy_data` folder. The codes can be tested on the data provided in the toy_data folder.

