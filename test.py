import numpy as np
from utils.utils import AverageMeter

#-------------------------------------------------------------------------------
def test_epoch(model, test_loader):
    prediction = np.array([])
    for batch_idx, (batch_data, batch_label) in enumerate(test_loader):
        batch_data = batch_data.cuda()
        batch_label = batch_label.cuda()   

        # predict the data without label
        batch_prediction = model(batch_data)

        # the maximum of the possibility 
        _, pred = batch_prediction.topk(1, 1, True, True)
        pred_squeeze = pred.squeeze()
        prediction = np.append(prediction, pred_squeeze.data.cpu().numpy())
    return prediction
#-------------------------------------------------------------------------------