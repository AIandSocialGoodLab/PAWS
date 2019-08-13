import os
import filecmp
my_path = 'C:\\Users\\MaxWillx\\CMU course\\Paws Project\\PAWS_SoftWare\\QgisIntegration\\csvfiles123'
std_path = 'C:\\Users\\MaxWillx\\CMU course\\Paws Project\\PAWS_SoftWare\\QgisIntegration\\csv_standard'
for file in os.listdir(std_path):
  std_file = os.path.join(std_path, file)
  my_file = os.path.join(my_path, file)
  if not os.path.isfile(my_file):
    raise Exception(std_file + ' no matching file')
  if filecmp.cmp(my_file, std_file):
    continue
  raise Exception(my_file + ' content not matched')
