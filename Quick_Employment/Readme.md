
### File Descriptions

In both automate_data.py and make_data_pandas.py, there is a chunk of code labeled "Module 0" that requires the specification of the data being used, whether in the format of a raster shapefile, a vector shapefile, or an excel spreadsheet. Variables should be self explanatory and the documenation within these files should also give decent descriptions.

automate_data.py - This file automatically creates an excel sheet with the "is-" features, "dist-" features, slope and elevation features, and normalized versions of these features when given shapefiles to draw data from.

dataset.py - Contains an implementation of the dataset object used to help facilitate implementation of make_data_pandas.py.

make_data_pandas.py - This file tests an ensembling method coupled with xgboost, dynamic negative sampling, and oversampling on a given set of data, producing the predicted poaching probabilities into a text file and creating a raster file to demonstrates the said predictions.

### Run Code
automate_data.py and make_data_pandas.py can be run using the following command.
```
python3 filename 
```
where the `filename` should be replaced with the name of the corresponding file to be run.

