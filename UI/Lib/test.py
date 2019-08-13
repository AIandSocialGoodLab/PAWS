import sys
import time
import os
import shutil
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PyQt5.QtWidgets import *
from PyQt5 import QtCore, QtGui, QtWidgets

from QgisIntegration.QgisStandalone import QgisStandalone
from run_makedata import run_model, save_model, extract_features


class CheckableComboBox(QComboBox):

  def __init__(self, parent=None):
    super(CheckableComboBox, self).__init__(parent)
    self.view().pressed.connect(self.handleItemPressed)
    self.setModel(QtGui.QStandardItemModel(self))

  def handleItemPressed(self, index):

    item = self.model().itemFromIndex(index)
    if item.checkState() == QtCore.Qt.Checked:
      item.setCheckState(QtCore.Qt.Unchecked)
      self.select_all_item.setCheckState(QtCore.Qt.Unchecked)
    else:
      item.setCheckState(QtCore.Qt.Checked)
    if self.select_all_item.checkState() == QtCore.Qt.Checked:
      for index in range(self.count()):
        item_ = self.model().item(index)
        item_.setCheckState(QtCore.Qt.Checked)
    return

  def checkDefaultFeats(self):
    self.select_all_item.setCheckState(QtCore.Qt.Unchecked)
    for index in range(1, self.count()):
      item_ = self.model().item(index)
      if 'normal' in item_.text():
        item_.setCheckState(QtCore.Qt.Checked)
    return

  def getCheckItem(self):
    # getCheckItem可以获得选择的项目text
    checkedItems = []
    for index in range(1, self.count()):
      item = self.model().item(index)
      if item.checkState() == QtCore.Qt.Checked:
        checkedItems.append(item.text())
    return checkedItems

  # def checkedItems(self):
  #     checkedItems = []
  #     for index in range(self.count()):
  #         item = self.model().item(index)
  #         if item.checkState() == QtCore.Qt.Checked:
  #             checkedItems.append(item)
  #     return checkedItems

  def addItems(self, itemList):
    self.addItem(' select all files')
    self.select_all_item = self.model().item(0, 0)
    self.select_all_item.setCheckState(QtCore.Qt.Unchecked)
    for index, element in enumerate(itemList):
      self.addItem(element)
      item = self.model().item(index + 1, 0)
      item.setCheckState(QtCore.Qt.Unchecked)
    return


class MainForm(QWidget):

  def __init__(self, name='MainForm'):
    super(MainForm, self).__init__()
    self.setWindowTitle(name)
    self.cwd = os.getcwd()
    self.resize(1000, 660)

    self.chosen_file = None
    self.preprocess_runable = [0, 0, 0, 0]
    self.preprocess_done = False
    self.chosen_model = None
    self.output = None
    self.save_path = None

    self.btn_chooseFile = QPushButton("Select input folder", self)
    self.btn_chooseFile.setGeometry(60, 60, 270, 60)
    self.label1 = QLabel('Please select input folder', self)
    self.label1.setGeometry(360, 60, 580, 60)

    self.btn_selectBoundary = QComboBox(self)
    self.btn_selectBoundary.setGeometry(60, 140, 270, 60)
    self.btn_selectBoundary.addItems(['        Select Boundary'])

    self.btn_selectPatrol = QComboBox(self)
    self.btn_selectPatrol.setGeometry(360, 140, 270, 60)
    self.btn_selectPatrol.addItems(['        Select Patrol'])

    self.btn_selectPoach = QComboBox(self)
    self.btn_selectPoach.setGeometry(660, 140, 270, 60)
    self.btn_selectPoach.addItems(['        Select Poaching'])

    self.btn_runPreprocess = QPushButton("Data preprocess", self)
    self.btn_runPreprocess.setGeometry(60, 220, 270, 60)

    self.btn_selectModel = QComboBox(self)
    self.btn_selectModel.setGeometry(60, 300, 270, 60)
    self.btn_selectModel.addItems(
        ['          Select model', '          XGBOOST', '          Decision Tree', '          SVM'])

    self.btn_runModel = QPushButton("Run model", self)
    self.btn_runModel.setGeometry(60, 380, 275, 60)

    self.btn_selectFeature = CheckableComboBox(self)
    self.btn_selectFeature.setGeometry(360, 380, 270, 60)
    self.btn_selectFeature.addItem('        Select Feature')

    self.btn_chooseDir = QPushButton("Select output folder", self)
    self.btn_chooseDir.setGeometry(60, 460, 270, 60)
    self.label2 = QLabel('Please select output folder', self)
    self.label2.setGeometry(360, 460, 580, 60)

    self.btn_exportResult = QPushButton("Save result", self)
    self.btn_exportResult.setGeometry(60, 540, 270, 60)

    self.btn_runPreprocess.setEnabled(False)
    self.btn_runModel.setEnabled(False)
    self.btn_exportResult.setEnabled(False)

    self.btn_chooseFile.clicked.connect(self.slot_btn_chooseFile)
    self.btn_runPreprocess.clicked.connect(self.slot_btn_runPreprocess)
    self.btn_selectBoundary.activated[
        str].connect(self.slot_btn_selectBoundary)
    self.btn_selectPatrol.activated[str].connect(self.slot_btn_selectPatrol)
    self.btn_selectPoach.activated[str].connect(self.slot_btn_selectPoach)
    self.btn_selectModel.activated[str].connect(self.slot_btn_selectModel)
    self.btn_runModel.clicked.connect(self.slot_btn_runModel)
    self.btn_chooseDir.clicked.connect(self.slot_btn_chooseDir)
    self.btn_exportResult.clicked.connect(self.slot_btn_exportResult)

  def slot_btn_chooseFile(self):
    self.chosen_file = QFileDialog.getExistingDirectory(
        self, "getExistingDirectory", "./")
    print('chosen_file', self.chosen_file)
    if self.chosen_file:
      self.label1.setText(self.chosen_file)
      files = os.listdir(self.chosen_file)
      files = list(set([i.split('.')[0] for i in files
                        if i.split('.')[1] in ["shp", "tif"]
                        ]))
      files.sort()
      files = ['        ' + i for i in files]
      self.btn_selectBoundary.clear()
      self.btn_selectPoach.clear()
      self.btn_selectPatrol.clear()
      self.btn_selectBoundary.addItems(['        Select Boundary'] + files)
      self.btn_selectPoach.addItems(['        Select Poaching'] + files)
      self.btn_selectPatrol.addItems(['        Select Patrol'] + files)
      self.preprocess_runable = [1, 0, 0, 0]
    else:
      self.label1.setText('Please select input folder')
      self.preprocess_runable = [0, 0, 0, 0]
      self.btn_selectBoundary.clear()
      self.btn_selectPoach.clear()
      self.btn_selectPatrol.clear()
      self.btn_selectBoundary.addItems(['        Select Boundary'])
      self.btn_selectPoach.addItems(['        Select Poach'])
      self.btn_selectPatrol.addItems(['        Select Patrol'])
    self.btn_runPreprocess.setEnabled(False)
    self.btn_runModel.setEnabled(False)
    self.btn_selectFeature.clear()
    self.btn_selectFeature.addItem('        Select Feature')
    self.btn_exportResult.setEnabled(False)
    self.preprocess_done = False
    self.boundary_name = None
    self.patrol_name = None
    self.poach_name = None
    return

  def slot_btn_selectBoundary(self, text):
    self.boundary_name = text.strip()
    if self.boundary_name != 'Select Boundary':
      self.preprocess_runable[1] = 1
    else:
      self.preprocess_runable[1] = 0
    if min(self.preprocess_runable) == 1:
      self.btn_runPreprocess.setEnabled(True)
    else:
      self.btn_runPreprocess.setEnabled(False)
      self.btn_runModel.setEnabled(False)
      self.btn_selectFeature.clear()
      self.btn_selectFeature.addItem('        Select Feature')
      self.btn_exportResult.setEnabled(False)
    return

  def slot_btn_selectPatrol(self, text):
    self.patrol_name = text.strip()
    if self.patrol_name != 'Select Patrol':
      self.preprocess_runable[2] = 1
    else:
      self.preprocess_runable[2] = 0
    if min(self.preprocess_runable) == 1:
      self.btn_runPreprocess.setEnabled(True)
    else:
      self.btn_runPreprocess.setEnabled(False)
      self.btn_runModel.setEnabled(False)
      self.btn_selectFeature.clear()
      self.btn_selectFeature.addItem('        Select Feature')
      self.btn_exportResult.setEnabled(False)
    return

  def slot_btn_selectPoach(self, text):
    self.poach_name = text.strip()
    if self.poach_name != 'Select Poaching':
      self.preprocess_runable[3] = 1
    else:
      self.preprocess_runable[3] = 0
    if min(self.preprocess_runable) == 1:
      self.btn_runPreprocess.setEnabled(True)
    else:
      self.btn_runPreprocess.setEnabled(False)
      self.btn_runModel.setEnabled(False)
      self.btn_selectFeature.clear()
      self.btn_selectFeature.addItem('        Select Feature')
      self.btn_exportResult.setEnabled(False)
    return

  def slot_btn_runPreprocess(self):

    self.btn_chooseFile.setEnabled(False)
    self.btn_selectModel.setEnabled(False)
    self.btn_runPreprocess.setEnabled(False)
    self.btn_chooseDir.setEnabled(False)
    # QMessageBox.information(self, 'info0', 'Running data preprocessing, please wait')

    qgis = QgisStandalone(
        qgis_boundary_file=self.boundary_name + '.shp',
        qgis_install_path="C:\\Program Files (x86)\\QGIS 2.18",
        qgis_input_shp_path=self.chosen_file,
        qgis_output_shapefile_path='tmp_shapefile',
        qgis_output_csv_path='tmp_csvfile'
    )
    qgis.run()
    self.final_data, _, self.selected_features = extract_features(
        'tmp_csvfile')
    self.selected_features = [f for f in self.selected_features
                              if self.poach_name not in f and self.patrol_name not in f]
    self.preprocess_done = True
    self.btn_chooseFile.setEnabled(True)
    self.btn_selectModel.setEnabled(True)
    self.btn_runPreprocess.setEnabled(True)
    self.btn_chooseDir.setEnabled(True)
    if self.chosen_model:
      self.btn_runModel.setEnabled(True)
      # Feature_list = sorted(os.listdir('tmp_csvfile'))
      # print('before', Feature_list)
      # Feature_list = [i.replace('.csv','') for i in Feature_list
      #               if i!='X.csv' and i!='Y.csv'
      #               and (self.poach_name.replace('.csv','') not in i) and (self.patrol_name.replace('.csv','') not in i)]
      # print('after', Feature_list)
      self.btn_selectFeature.clear()
      self.btn_selectFeature.addItems(self.selected_features)
      self.btn_selectFeature.checkDefaultFeats()
    else:
      self.btn_runModel.setEnabled(False)
      self.btn_selectFeature.clear()
      self.btn_selectFeature.addItem('        Select Feature')
    return

  def slot_btn_selectModel(self, text):
    mapping = {'          Select model': None, '          XGBOOST': 'xgb',
               '          Decision Tree': 'dt', '          SVM': 'svm'}
    self.chosen_model = mapping[text]
    if self.preprocess_done and self.chosen_model:
      self.btn_runModel.setEnabled(True)
      # Feature_list = sorted(os.listdir('tmp_csvfile'))
      # Feature_list = [' '+ i.replace('.csv','') for i in Feature_list if i!='X.csv' and i!='Y.csv']
      self.btn_selectFeature.clear()
      self.btn_selectFeature.addItems(self.selected_features)
      self.btn_selectFeature.checkDefaultFeats()
    else:
      self.btn_runModel.setEnabled(False)
      self.btn_selectFeature.clear()
      self.btn_selectFeature.addItem('        Select Feature')
    return

  def slot_btn_runModel(self):
    mapping = {'xgb': 'XGBOOST', 'dt': 'Decision Tree', 'svm': 'SVM'}
    # QMessageBox.information(self, 'info1', 'Running {}, please wait'.format(mapping[self.chosen_model]))

    self.btn_runModel.setEnabled(False)
    self.btn_chooseFile.setEnabled(False)
    self.btn_selectModel.setEnabled(False)
    features = self.btn_selectFeature.getCheckItem()
    print('runModel| checked features:', features)
    self.output = run_model('tmp_csvfile',
                            selected_features=features,
                            final_data=self.final_data,
                            method=self.chosen_model,
                            patrol_name=self.patrol_name,
                            poaching_name=self.poach_name)

    # shutil.rmtree
    self.btn_runModel.setEnabled(True)
    self.btn_chooseFile.setEnabled(True)
    self.btn_selectModel.setEnabled(True)
    if self.output:
      QMessageBox.information(
          self, 'info2', 'Running {} finished'.format(mapping[self.chosen_model]))
      if self.save_path:
        self.btn_exportResult.setEnabled(True)
    else:
      QMessageBox.information(
          self, 'info3', 'Running {} failed'.format(mapping[self.chosen_model]))
    return

  def slot_btn_chooseDir(self):
    self.save_path = QFileDialog.getExistingDirectory(
        self, "getExistingDirectory", "./")
    if self.save_path and self.output:
      self.label2.setText(self.save_path)
      self.btn_exportResult.setEnabled(True)
    else:
      self.label2.setText('Select folder')
      self.btn_exportResult.setEnabled(False)
    return

  def slot_btn_exportResult(self):
    yea, mon, day, hou, minu, sec = list(time.localtime())[:6]
    name = '/PAWS%d_%02d_%02d_%02d_%02d_%02d.asc' % (
        yea, mon, day, hou, minu, sec)
    save_model(self.output, self.save_path + name)
    QMessageBox.information(
        self, 'info4', 'Results are saved in \n{}'.format(self.save_path))
    pic = np.loadtxt(self.save_path + name, skiprows=6)

    pic[pic == 0] = -1

    plt.imshow(pic)
    yellow = mpatches.Patch(color='#ffff00', label='high freq')
    green = mpatches.Patch(color='#009999', label='low freq')
    plt.legend(handles=[yellow, green])
    try:
      plt.savefig(self.save_path + name.replace('asc', 'png'))
    except:
      pass
    return

  def closeEvent(self, event):
    reply = QMessageBox.question(self,
                                 'exit',
                                 "Do you want to exit?",
                                 QMessageBox.Yes | QMessageBox.No,
                                 QMessageBox.No)
    if reply == QMessageBox.Yes:
      try:
        shutil.rmtree('tmp_shapefile')
        shutil.rmtree('tmp_csvfile')
        shutil.rmtree('processing')
      except:
        pass
      event.accept()
    else:
      event.ignore()


if __name__ == "__main__":
  try:
    shutil.rmtree('tmp_shapefile')
    shutil.rmtree('tmp_csvfile')
    shutil.rmtree('processing')
  except:
    pass
  app = QApplication(sys.argv)
  mainForm = MainForm('PAWS')
  mainForm.show()
  sys.exit(app.exec_())
