This is a software based on Python and QGIS 2.6 to predict poaching signs. Tutorial could be found in the QucikEmployment_Toy folder.

### File Descriptions
Below are brief descriptions of the files included in the folder QuickEmployment_Toy. 

automate_data.py - This file automatically creates an excel sheet with the features and their normalized versions when given shapefiles to draw data from.

dataset.py - Contains an implementation of the dataset object used to help facilitate implementation of make_data_pandas.py.

make_data_pandas.py - This file tests an ensembling method coupled with xgboost, dynamic negative sampling, and oversampling on a given set of data, producing the predicted poaching probabilities into a text file and creating a raster file to demonstrates the said predictions. Can also create predictions for unlabeled region given the same feature space for a labeled conservation site.

PAWS_tutorial.pdf - detailed instructions on how to set up and use these tools.

