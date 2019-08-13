import subprocess
import os
import shutil


class QgisStandalone(object):
  """
  Provide constructor with qgis install path, input .shp path, input layers selection, output shapefiles path, output .csv path
  The path format is provided as follows, you should initialize this object by providing your own path.
  Note that QgisStandalone.py and automate_data.py should be in the same folder, bash file is also generated in module folder
  """

  def __init__(self, **args):
    super(QgisStandalone, self).__init__()

    self.qgis_install_path = args['qgis_install_path']
    self.qgis_sub_install_path = os.path.join(
        self.qgis_install_path, 'apps', 'qgis-ltr')
    self.qgis_env_bat_path = os.path.join(
        self.qgis_install_path, 'bin', 'o4w_env.bat')

    self.qgis_input_shp_path = args['qgis_input_shp_path'].replace("\\", "/")
    self.qgis_output_shapefile_path = args[
        'qgis_output_shapefile_path'].replace("\\", "/")
    self.qgis_output_csv_path = args['qgis_output_csv_path'].replace("\\", "/")

    self.qgis_bash_path = os.path.join(os.path.dirname(
        os.path.realpath(__file__)), 'env_test.bat')
    self.qgis_automate_data_path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), 'automate_data.py')

    self.qgis_input_layers = None
    if 'qgis_input_layers' in args:  # optional. dictionary specify which layer to process
      self.qgis_input_layers = args['qgis_input_layers']

    self.serialize_file = os.path.join(os.path.dirname(
        os.path.realpath(__file__)), 'layer_file_name.txt')
    self.default_layer_name = ['boundary_file',
                               'dist_layers', 'int_layers', 'raster_layers']
    self.qgis_boundary_file = args['qgis_boundary_file']

    if self.qgis_input_layers is None:
      self.qgis_input_layers = self.default_select()

    self.layer_name = list(self.qgis_input_layers.keys())

  def run(self):
    self.make_dir(self.qgis_output_csv_path, self.qgis_output_shapefile_path)
    qgis_bash_script = self.serialize_bash_script(self.qgis_bash_path)
    self.serialize_layer_name()
    self.check_path()
    self.clear_output_path(
        self.qgis_output_shapefile_path, self.qgis_output_csv_path)

    FNULL = open(os.devnull, 'w')
    retcode = subprocess.call(qgis_bash_script, stderr=FNULL)
    print('Subprocess finished with exit code ' + str(retcode))
    os.remove(self.qgis_bash_path)
    os.remove(self.serialize_file)
    if retcode:
      if retcode == 3221225477:
        pass  # qgis warnings in standalone evironment, ignored
      else:
        raise Exception('Qgis Internal Error')

  def list_files(self, path):
    if not os.path.exists(path):
      raise Exception('Qgis .shp input path: "' + path + '" not found')
    return os.listdir(path)

  def default_select(self):
    layer_name = dict()
    for field_name in self.default_layer_name:
      layer_name[field_name] = list()
    input_files = self.list_files(self.qgis_input_shp_path)
    for file in input_files:
      if os.path.splitext(file)[1] == '.shp':
        if file == self.qgis_boundary_file:
          layer_name['boundary_file'].append(file)
          continue
        layer_name['dist_layers'].append(file)
        layer_name['int_layers'].append(file)
      elif os.path.splitext(file)[1] == '.tif':
        layer_name['raster_layers'].append(file)
      else:
        continue
    return layer_name

  def clear_output_path(self, *paths):
    for path in paths:
      files = os.listdir(path)
      for file in files:
        try:
          absolute_path = os.path.join(path, file)
          if os.path.isfile(absolute_path):
            os.remove(absolute_path)
          elif os.path.isdir(absolute_path):
            shutil.rmtree(absolute_path)
        except Exception as e:
          print(e)
      files = os.listdir(path)
      if files:
        raise Exception('can\'t remove file')

  def serialize_layer_name(self):
    with open(self.serialize_file, 'w') as f:
      for layer in self.layer_name:
        f.write(layer + ': ')
        file_list = self.qgis_input_layers[layer]
        for i in range(len(file_list)):
          if i == len(file_list) - 1:
            f.write(file_list[i])
            continue
          f.write(file_list[i] + ', ')
        if layer == self.layer_name[-1]:
          continue
        f.write('\n')

  def make_dir(self, *args):
    for path in args:
      if not os.path.exists(path):
        os.mkdir(path)

  def check_path(self):

    if not os.path.exists(self.qgis_install_path):
      raise Exception('Qgis Install Path: "' + self.qgis_install_path +
                      '" not found, make sure you have installed QGIS V2.18-ltr')

    if not os.path.exists(self.qgis_sub_install_path):
      raise Exception('Qgis Sub Install Path: "' + self.qgis_sub_install_path +
                      '" not found, make sure you have installedrr QGIS V2.18-ltr')

    if not os.path.exists(self.qgis_env_bat_path):
      raise Exception('Qgis env_bat: "' +
                      self.qgis_env_bat_path + '" not found')

    if not os.path.isfile(self.qgis_automate_data_path):
      raise Exception('automate_data.py file: "' +
                      self.qgis_automate_data_path + '" not found')

    if not os.path.exists(self.qgis_input_shp_path):
      raise Exception('Qgis .shp input path: "' +
                      self.qgis_input_shp_path + '" not found')

    if not os.path.exists(self.qgis_output_shapefile_path):
      raise Exception('Qgis shapefile output path "' +
                      self.qgis_output_shapefile_path + '" not found')

    if not os.path.exists(self.qgis_output_csv_path):
      raise Exception('Qgis output csv path: "' +
                      self.qgis_output_csv_path + '" not found')

    if not os.path.isfile(self.qgis_bash_path):
      raise Exception('Qgis bash script file: "' +
                      self.qgis_bash_path + '" not found')

    for key in self.layer_name:
      layer_files = self.qgis_input_layers[key]
      if not layer_files:
        if key == 'boundary_file':
          raise Exception('make sure ' + self.qgis_boundary_file +
                          ' is in ' + self.qgis_input_shp_path)
        if key == 'raster_layers':
          continue
        raise Exception('missing layer type: ' + key)

      for layer_file in layer_files:
        absolute_path = os.path.join(
            self.qgis_input_shp_path.replace('/', '\\'), layer_file)
        if not os.path.isfile(absolute_path):
          raise Exception(absolute_path + ' not found')

  def serialize_bash_script(self, path):
    """
    write QGIS bash script:
    1. setup qgis standalone envrionment
    2. setup qgis data input folder
    3. setup qgis data output folder
    """
    buffer = []
    buffer.append("SET OSGEO4W_ROOT=" + self.qgis_install_path + "\n")
    buffer.append("SET QGISNAME=qgis-ltr\n")
    buffer.append("SET QGIS=%s\n" % self.qgis_sub_install_path)
    buffer.append("SET QGIS_PREFIX_PATH=%QGIS%\n")
    # buffer.append("echo %PATH%\r\n")
    buffer.append("CALL \"%s\"\n" % self.qgis_env_bat_path)
    buffer.append("SET PATH=%PATH%;%QGIS%\\bin\n")
    buffer.append("SET PYTHONPATH=%QGIS%\\python;%PYTHONPATH%\n")
    buffer.append("SET PYTHONPATH=" + self.qgis_sub_install_path +
                  "\\python\\plugins;%PYTHONPATH%\n")
    buffer.append("SET PYTHONPATH=" +
                  self.qgis_sub_install_path + "\\plugins;%PYTHONPATH%\n")
    buffer.append("SET PYTHONPATH=" +
                  self.qgis_sub_install_path + "\\python;%PYTHONPATH%\n")
    # buffer.append("echo %PATH%\r\n")
    buffer.append('python "%s" "%s/" "%s/" "%s/" "%s"\n' % (
        self.qgis_automate_data_path, self.qgis_input_shp_path,
        self.qgis_output_shapefile_path, self.qgis_output_csv_path,
        self.serialize_file))

    with open(path, "w") as f:
      print("Saving bash script to " + path)
      f.write(''.join(buffer))

    return path

if __name__ == "__main__":
  qgis_standalone = QgisStandalone()
  qgis_standalone.run()
