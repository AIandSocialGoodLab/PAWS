
### File Descriptions

In both automate_data.py and make_data_pandas.py, there is a chunk of code labeled "Module 0" that requires the specification of the data being used, whether in the format of a raster shapefile, a vector shapefile, or an excel spreadsheet. Variables should be self explanatory and the documentation within these files should also give decent descriptions.

automate_data.py - This file automatically creates an excel sheet with the features and their normalized versions when given shapefiles to draw data from.

dataset.py - Contains an implementation of the dataset object used to help facilitate implementation of make_data_pandas.py.

make_data_pandas.py - This file tests an ensembling method coupled with xgboost, dynamic negative sampling, and oversampling on a given set of data, producing the predicted poaching probabilities into a text file and creating a raster file to demonstrates the said predictions. Can also create predictions for unlabeled region given the same feature space for a labeled conservation site.

tutorial.pdf - Some more in depth instructions on how to use these tools.

### Run Code
automate_data.py and make_data_pandas.py can be run using the following command.
```
python3 filename 
```
where the `filename` should be replaced with the name of the corresponding file to be run.

### Important Note:
Code in QuickEmployment_Toy folder has been modified to remove pandas dependency in automate_data.py. The pandas dependent code has been relocated to make_data_pandas.py. PAWS_tutorial.pdf is a tutorial for the code in QuickEmployment_Toy folder. To run the tool, please download QuickEmployment_Toy folder and follow PAWS_tutorial.pdf  

