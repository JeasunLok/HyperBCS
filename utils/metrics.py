import numpy as np
from sklearn.metrics import confusion_matrix

# calculate the accuracy of the prediction using label (with function topk)
#-------------------------------------------------------------------------------
def accuracy(output, label, topk=(1,)):
  maxk = max(topk)
  batch_size = label.size(0)

  _, pred = output.topk(maxk, 1, True, True)
  pred = pred.t()
  correct = pred.eq(label.view(1, -1).expand_as(pred))

  res = []
  for k in topk:
    correct_k = correct[:k].view(-1).float().sum(0)
    res.append(correct_k.mul_(100.0/batch_size))
  return res, label, pred.squeeze()
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
def output_metric(label, prediction):
    CM = confusion_matrix(label, prediction)
    OA, AA, Kappa, CA = cal_results(CM)
    return OA, AA, Kappa, CA, CM
#-------------------------------------------------------------------------------

# calculating the OA(overall accuracy), AA(average accuracy), Kappa, CA(class accuracy)
#-------------------------------------------------------------------------------
def cal_results(matrix):
    shape = np.shape(matrix)
    number = 0
    sum = 0
    CA = np.zeros([shape[0]], dtype=np.float64)
    for i in range(shape[0]):
        number += matrix[i, i]
        CA[i] = matrix[i, i] / np.sum(matrix[i, :])
        sum += np.sum(matrix[i, :]) * np.sum(matrix[:, i])
    OA = number / np.sum(matrix)
    AA = np.mean(CA)
    PE = sum / (np.sum(matrix) ** 2)
    Kappa = (OA - PE) / (1 - PE)
    return OA, AA, Kappa, CA
#-------------------------------------------------------------------------------