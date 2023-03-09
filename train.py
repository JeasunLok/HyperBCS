import numpy as np
from tqdm import tqdm
from utils.utils import AverageMeter
from utils.metrics import accuracy

#-------------------------------------------------------------------------------
# train model
def train_epoch(model, train_loader, criterion, optimizer, e, epoch):
    loss_show = AverageMeter()
    acc = AverageMeter()
    label = np.array([])
    prediction = np.array([])
    loop = tqdm(enumerate(train_loader), total = len(train_loader))
    for batch_idx, (batch_data, batch_label) in loop:
        batch_data = batch_data.cuda()
        batch_label = batch_label.cuda()   

        optimizer.zero_grad()
        batch_prediction = model(batch_data)
        loss = criterion(batch_prediction, batch_label)
        loss.backward()
        optimizer.step()       

        # calculate the accuracy
        acc_batch, l, p = accuracy(batch_prediction, batch_label, topk=(1,))
        n = batch_data.shape[0]

        # update the loss and the accuracy 
        loss_show.update(loss.data, n)
        acc.update(acc_batch[0].data, n)
        label = np.append(label, l.data.cpu().numpy())
        prediction = np.append(prediction, p.data.cpu().numpy())

        loop.set_description(f'Train Epoch [{e}/{epoch}]')
        loop.set_postfix({"train_loss":loss_show.average.item(),
                          "train_accuracy": acc.average.item()})

    return acc.average, loss_show.average, label, prediction
#-------------------------------------------------------------------------------

# validate model
def valid_epoch(model, valid_loader, criterion, optimizer):
    loss_show = AverageMeter()
    acc = AverageMeter()
    label = np.array([])
    prediction = np.array([])
    loop = tqdm(enumerate(valid_loader), total = len(valid_loader))
    for batch_idx, (batch_data, batch_label) in loop:
        batch_data = batch_data.cuda()
        batch_label = batch_label.cuda()   

        batch_prediction = model(batch_data)
        loss = criterion(batch_prediction, batch_label)

        # calculate the accuracy
        acc_batch, l, p = accuracy(batch_prediction, batch_label, topk=(1,))
        n = batch_data.shape[0]

        # update the loss and the accuracy 
        loss_show.update(loss.data, n)
        acc.update(acc_batch[0].data, n)
        label = np.append(label, l.data.cpu().numpy())
        prediction = np.append(prediction, p.data.cpu().numpy())

        loop.set_description(f'Val Epoch')
        loop.set_postfix({"val_loss":loss_show.average.item(),
                          "val_accuracy": acc.average.item()})
        
    return label, prediction
#-------------------------------------------------------------------------------