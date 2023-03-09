import numpy as np
from tqdm import tqdm
from utils.utils import AverageMeter

#-------------------------------------------------------------------------------
def test_epoch(model, test_loader):
    prediction = np.array([])
    loop = tqdm(enumerate(test_loader), total = len(test_loader))
    for batch_idx, (batch_data, batch_label) in loop:
        batch_data = batch_data.cuda()
        batch_label = batch_label.cuda()   

        # predict the data without label
        batch_prediction = model(batch_data)

        # the maximum of the possibility 
        _, pred = batch_prediction.topk(1, 1, True, True)
        pred_squeeze = pred.squeeze()
        prediction = np.append(prediction, pred_squeeze.data.cpu().numpy())

        loop.set_description(f'Test Epoch')  
    return prediction
#-------------------------------------------------------------------------------