from scipy.io import loadmat
import numpy as np
import random
import math
import matplotlib.pyplot as plt

# a class for calculating the average of the accuracy and the loss
#-------------------------------------------------------------------------------
class AverageMeter(object):

  def __init__(self):
    self.reset()

  def reset(self):
    self.average = 0 
    self.sum = 0
    self.count = 0

  def update(self, val, n=1):
    self.sum += val * n
    self.count += n
    self.average = self.sum / self.count
#-------------------------------------------------------------------------------

# a function to load our wetland data
def load_wetland_data(time_folder, image_path, label_path, wetland_id, mode, value, save_colormap):
  data_year = str(wetland_id)[-2:]
  image = loadmat(image_path)
  label = loadmat(label_path)
  image_mark = "image_" + data_year
  label_mark = "label_" + data_year
  image = image[image_mark]
  label = label[label_mark]
  train_label = np.zeros(label.shape)
  test_label = np.copy(label)
  num_classes = int(np.max(label))
  for i in range(num_classes):
    sample = np.array([])
    train_sample_num = 0
    num = sum(sum(label==(i+1)))
    if(mode == "percentage"):
      train_sample_num = math.floor(num*value)
      sample = random.sample(range(1,num), train_sample_num)
    elif(mode == 'fixed'):
      train_sample_num = math.floor(value)
      sample = random.sample(range(1,num), train_sample_num)
    position = np.argwhere(label==(i+1))
    sample_position=position[sample]
    for j in range(math.floor(train_sample_num)):
      train_label[sample_position[j][0]][sample_position[j][1]] = label[sample_position[j][0]][sample_position[j][1]]
      test_label[sample_position[j][0]][sample_position[j][1]] = 0

  # save the image of train and test samples
  plt.imsave(time_folder + r"\\all_label.png", label, cmap=save_colormap, dpi=300)
  plt.imsave(time_folder + r"\\train_label.png", train_label, cmap=save_colormap, dpi=300)
  plt.imsave(time_folder + r"\\test_label.png", test_label, cmap=save_colormap, dpi=300)
  return image, train_label, test_label
