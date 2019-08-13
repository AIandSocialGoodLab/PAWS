import make_data_pandas

from mydataset import DataSet
from os import listdir, makedirs
from os.path import isfile, join, exists, splitext


def get_csv_files_in_dir(base_path):
  '''get csv files in dir'''
  files = [f for f in listdir(base_path)
           if isfile(join(base_path, f)) and '.csv' == splitext(f)[-1]]
  return files


def main_get_final_data(basepath, input_files):
  # remove ".csv"
  column_names = [f[:-4] for f in input_files]

  files = []
  for name in input_files:
    files.append(join(basepath, name))

  # final_data: originally: final.csv
  return make_data_pandas.process_automate_data(files, column_names)


def main_predict(final_data, selected_features, patrol, poaching,
                 method='xgb', ratio=100,):
  df_alldata, df_invaliddata, df_unknowndata, df_allpositive, \
      df_allnegative, df_slct_positive, df_slct_negative, \
      df_slct_unlabeled, \
      PositiveData, NegativeData, UnknownData = \
      make_data_pandas.preprocessing_fn1(
          final_data, patrol, poaching, selected_features)

  df_alldata2, df_validdata2, df_invaliddata2, df_slct_valid, NewAllData = \
      make_data_pandas.preprocessing_fn2(final_data, selected_features)

  neg, NegativeData, NotFam, neg_label, Fam, dataset = \
      make_data_pandas.build_dataset(
          PositiveData, NegativeData, UnknownData, FoldNum=4)

  # name of text file output for probabilistic predictions of
  # each grid cell in conservations 1 and 2
  qgis_file_in1 = "predictions1.txt"
  qgis_file_in2 = "predictions2.txt"
  qgis_file_in1_str = make_data_pandas.main_poaching_predict(
      qgis_file_in1,
      qgis_file_in2,
      df_allpositive,
      df_allnegative,
      df_unknowndata,
      df_validdata2,
      df_slct_positive,
      df_slct_negative,
      df_slct_unlabeled,
      df_slct_valid,
      dataset,
      df_invaliddata,
      df_invaliddata2,
      method,
      ratio,
  )
  return df_alldata, qgis_file_in1_str


def main_prep_qgis(qgis_file_in1_str, df_alldata,
                   qgis_file_out1="predictions_heatmap1.asc"):
  # qgis_file_out1: raster file of probabilistic predictions

  # represents the coordinates of the left bottom corner for
  # conservation site 1 (longitude and latitude if working with WGS84)
  xcorner1 = 127.76402335
  ycorner1 = 43.5257568717

  # define the grid sizes (discretization levels) for each conservations ite
  # which should match from the automate_data.py script
  gridDim1 = 0.01
  make_data_pandas.prep_qgis(qgis_file_in1_str, qgis_file_out1, gridDim1,
                             xcorner1, ycorner1, df_alldata)


def extract_features(basepath):
  # Step-1, user tells us basepath, and then we get all the csv files
  # Must include all the input files from automate_data.py
  input_files = get_csv_files_in_dir(basepath)
  input_files.sort()

  # final_data: originally: final.csv
  # feature_names: a list of feature names
  final_data, feature_names = main_get_final_data(basepath, input_files)

  # Step-2, user tells us what features are prefered among selected_features
  # specify which features to use from final.csv feature spreadsheet
  selected_features = [f for f in feature_names
                       if (f not in ['normal-X', 'normal-Y'])
                       and (f.startswith('normal-') or f.startswith('is-'))]
  return final_data, feature_names, selected_features


def run_model(basepath, selected_features, final_data, method, patrol_name, poaching_name):
  # final_data, _, selected_features = extract_features(basepath)

  # specify which feature symbolizes where patrolling occurs
  patrol = 'is-' + patrol_name
  # specify which feature symbolizes where poaching occurs
  poaching = 'is-' + poaching_name

  print('run_model|patrol: ', patrol)
  print('run_model|poaching: ', poaching)

  # Step-3 run algorithm
  df_alldata, qgis_file_in1_str = main_predict(
      final_data, selected_features, patrol, poaching, method)

  output = (qgis_file_in1_str, df_alldata)
  return output


def save_model(output, qgis_file_out1):

  qgis_file_in1_str, df_alldata = output
  # qgis_file_out1: raster file of probabilistic predictions

  # represents the coordinates of the left bottom corner for
  # conservation site 1 (longitude and latitude if working with WGS84)
  xcorner1 = 127.76402335
  ycorner1 = 43.5257568717

  # define the grid sizes (discretization levels) for each conservations ite
  # which should match from the automate_data.py script
  gridDim1 = 0.01
  make_data_pandas.prep_qgis(qgis_file_in1_str, qgis_file_out1, gridDim1,
                             xcorner1, ycorner1, df_alldata)
  return


if __name__ == '__main__':
  main_real()
