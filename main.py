import os
import time
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.utils.data as Data
from scipy.io import loadmat,savemat

from models.vit_pytorch import ViT
from models.other_models import MLP_4,CNN_1D,CNN_2D,CNN_3D,CNN_3D_Classifer_1D,RNN_1D
from utils.data_processing import position_train_and_test_point,mirror_hsi,train_and_test_data,train_and_test_label
from utils.data_preparation import HSI_Dataset
from utils.metrics import output_metric
from utils.result_store_draw import draw_result_visualization,store_result
from utils.utils import load_wetland_data
from train import train_epoch,valid_epoch
from test import test_epoch

#-------------------------------------------------------------------------------
# setting the parameters
model_type = "Transformer" # CNN or Transformer
Transformer_mode = "CAF" # if Transformer : VIT CAF
CNN_mode = "CNN_2D" # if CNN : MLP_4 CNN_1D CNN_2D CNN_3D CNN_3D_Classifer_1D CNN_FCN RNN_1D
gpu = 0
epoch = 1000
test_freq = 100
batch_size = 128
patches = 3
band_patches = 3
learning_rate = 5e-4
weight_decay = 5e-3
gamma = 0.9

sample_mode = "fixed" # fixed or percentage
sample_value = 200 # fixed => numble of samples(int)  percentage => percentage of samples(0-1) 
HSI_data = "wetland" # IndianPine or wetland
year = 2015 # if wetland
#-------------------------------------------------------------------------------

# make the run folder in logs
#-------------------------------------------------------------------------------
time_now = time.localtime()
if model_type == "Transformer":
    if HSI_data == "IndianPine":
        time_folder = r".\\logs\\" + time.strftime("%Y-%m-%d-%H-%M-%S", time_now) + "-" + model_type + "-" + Transformer_mode + "-" + HSI_data
    else:
        time_folder = r".\\logs\\" + time.strftime("%Y-%m-%d-%H-%M-%S", time_now) + "-" + model_type + "-" + Transformer_mode + "-" + HSI_data + str(year)
elif model_type == "CNN_RNN":
    if HSI_data == "IndianPine":
        time_folder = r".\\logs\\" + time.strftime("%Y-%m-%d-%H-%M-%S", time_now) + "-" + model_type + "-" + CNN_mode + "-" + HSI_data
    else:
        time_folder = r".\\logs\\" + time.strftime("%Y-%m-%d-%H-%M-%S", time_now) + "-" + model_type + "-" + CNN_mode + "-" + HSI_data + str(year)
os.makedirs(time_folder)
#-------------------------------------------------------------------------------

# time setting
np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed(1)

# GPU settings
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
cudnn.deterministic = True
cudnn.benchmark = False

# IndianPine data loading
#-------------------------------------------------------------------------------
if HSI_data == "IndianPine":
    # color settings
    color_mat = loadmat(r".\\data\\AVIRIS_colormap.mat")
    color_mat_list = list(color_mat)
    color_matrix = color_mat[color_mat_list[3]] 

    # data loading
    data = loadmat(r".\\data\\IndianPine.mat")
    train_label = data['TR']
    test_label = data['TE']
    input = data['input'] #(145,145,200)
#-------------------------------------------------------------------------------

# wetland data
#-------------------------------------------------------------------------------
elif HSI_data == "wetland":
    # color settings
    colormap_mat = loadmat(r".\\data\\wetland_colormap.mat")
    colormap = colormap_mat["colormap_" + str(year)]
    colormap_1 = np.append("#FFFFFF", colormap)
    save_colormap_1 = mpl.colors.LinearSegmentedColormap.from_list('cmap', colormap_1.tolist(), 256)
    if model_type == "CNN_RNN":
        save_colormap_2 = save_colormap_1
    elif model_type == "Transformer":
        save_colormap_2 = mpl.colors.LinearSegmentedColormap.from_list('cmap', colormap.tolist(), 256)

    # data loading
    image_path = r".\\data\\" + str(year)[-2:] +"_image.mat"
    label_path = r".\\data\\" + str(year)[-2:] +"_label.mat"
    input, train_label, test_label = load_wetland_data(time_folder, image_path, label_path, year, sample_mode, sample_value, save_colormap_1)
#-------------------------------------------------------------------------------

# all_data and classes
all_label = train_label + test_label
num_classes = int(np.max(train_label))

# normalize data by band norm
input_normalize = np.zeros(input.shape)
for i in range(input.shape[2]):
    input_max = np.max(input[:,:,i])
    input_min = np.min(input[:,:,i])
    input_normalize[:,:,i] = (input[:,:,i]-input_min)/(input_max-input_min)
# data size
height, width, band = input.shape

# Transformer models
#-------------------------------------------------------------------------------
# obtain train and test data
if model_type == "Transformer":
    total_pos_train, total_pos_test, total_pos_true, number_train, number_test, number_true = position_train_and_test_point(train_label, test_label, all_label, num_classes)
    mirror_image = mirror_hsi(height, width, band, input_normalize, patch=patches)
    x_train_band, x_test_band, x_true_band = train_and_test_data(mirror_image, band, total_pos_train, total_pos_test, total_pos_true, patch=patches, band_patch=band_patches)
    y_train, y_test, y_true = train_and_test_label(number_train, number_test, number_true, num_classes)

    # data processing
    x_train=torch.from_numpy(x_train_band.transpose(0,2,1)).type(torch.FloatTensor) 
    y_train=torch.from_numpy(y_train).type(torch.LongTensor)
    train_dataset=Data.TensorDataset(x_train,y_train)
    train_loader=Data.DataLoader(train_dataset,batch_size=batch_size,shuffle=True)

    x_test=torch.from_numpy(x_test_band.transpose(0,2,1)).type(torch.FloatTensor) 
    y_test=torch.from_numpy(y_test).type(torch.LongTensor) 
    test_dataset=Data.TensorDataset(x_test,y_test)
    test_loader=Data.DataLoader(test_dataset,batch_size=batch_size,shuffle=True)

    x_true=torch.from_numpy(x_true_band.transpose(0,2,1)).type(torch.FloatTensor)
    y_true=torch.from_numpy(y_true).type(torch.LongTensor)
    true_dataset=Data.TensorDataset(x_true,y_true)
    true_loader=Data.DataLoader(true_dataset,batch_size=batch_size,shuffle=False)

    # create model
    model = ViT(
        image_size = patches,
        near_band = band_patches,
        num_patches = band,
        num_classes = num_classes,
        dim = 64,
        depth = 5,
        heads = 4,
        mlp_dim = 8,
        dropout = 0.1,
        emb_dropout = 0.1,
        mode = Transformer_mode
    )
#-------------------------------------------------------------------------------

# CNN models
#-------------------------------------------------------------------------------
elif model_type == "CNN_RNN":
    if CNN_mode == "MLP_4":
        model = MLP_4(
            input_channels = band,
            num_classes = num_classes + 1,
            dropout = True
        )
        patches = 1

    elif CNN_mode == "CNN_1D":
        model = CNN_1D(
            input_channels = band,
            num_classes = num_classes + 1
        )
        patches = 1

    elif CNN_mode == "CNN_2D":
        model = CNN_2D(
            input_channels = band,
            num_classes = num_classes + 1,
            patch_size = 64
        )
        patches = 64

    elif CNN_mode == "CNN_3D":
        model = CNN_3D(
            input_channels = band,
            num_classes = num_classes + 1,
            patch_size = 5,
            n_planes =  2
        )
        patches = 5

    elif CNN_mode == "CNN_3D_Classifer_1D":
        model = CNN_3D_Classifer_1D(
            input_channels = band,
            num_classes = num_classes + 1,
            patch_size = 5,
            dilation =  1
        )
        patches = 5
    
    elif CNN_mode == "RNN_1D":
        model = RNN_1D(
            input_channels = band,
            num_classes = num_classes + 1,
        )
        patches = 1
        
    train_dataset = HSI_Dataset(input_normalize, train_label, True, patches)
    train_loader = Data.DataLoader(train_dataset, batch_size, shuffle = True)

    test_dataset = HSI_Dataset(input_normalize, test_label, True, patches)
    test_loader = Data.DataLoader(test_dataset, batch_size, shuffle = True)

    true_dataset = HSI_Dataset(input_normalize, test_label, False, patches)
    true_loader = Data.DataLoader(true_dataset, batch_size, shuffle = False)

    total_pos_true = true_dataset.indices
#-------------------------------------------------------------------------------
model = model.cuda()
# criterion
criterion = nn.CrossEntropyLoss().cuda()
# optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
# scheduler
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=epoch//10, gamma=gamma)

# train
print("start training")
tic = time.time()
epoch_loss = np.zeros([2, epoch])
for e in range(epoch): 
    model.train()
    train_acc, train_loss, label_t, prediction_t = train_epoch(model, train_loader, criterion, optimizer)
    scheduler.step()
    OA_train, AA_train, Kappa_train, CA_train, CM_train = output_metric(label_t, prediction_t) 
    print("Epoch: {:03d} | train_loss: {:.4f} | train_acc: {:.4f}".format(e+1, train_loss, train_acc))
    epoch_loss[0][e], epoch_loss[1][e] = e+1, train_loss

    if ((e+1) % test_freq == 0) | (e == epoch - 1):         
        model.eval()
        label_v, prediction_v = valid_epoch(model, test_loader, criterion, optimizer)
        OA_val, AA_val, Kappa_val, CA_val, CM_val = output_metric(label_v, prediction_v)
        if (e != epoch -1):
            print("===============================================================================")
            print("Epoch: {:03d}  =>  OA: {:.4f} | AA: {:.4f} | Kappa: {:.4f}".format(e+1, OA_val, AA_val, Kappa_val))
            print("===============================================================================")

toc = time.time()
print("Running Time: {:.2f}".format(toc-tic))
print("end training")
print("===============================================================================")
print("Final result:")
print("OA: {:.4f} | AA: {:.4f} | Kappa: {:.4f}".format(OA_val, AA_val, Kappa_val))
print("CA:", end="")
print(CA_val)
print("Confusion Matrix:")
print(CM_val)
print("===============================================================================")

print("start testing")
model.eval()

# output classification maps
pre_u = test_epoch(model, true_loader)
prediction = np.zeros((height, width), dtype=float)
for i in range(total_pos_true.shape[0]):
    if model_type == "Transformer":
        prediction[total_pos_true[i,0], total_pos_true[i,1]] = pre_u[i] + 1
    elif model_type == "CNN_RNN":
        prediction[total_pos_true[i,0], total_pos_true[i,1]] = pre_u[i]

print("end testing")
print("===============================================================================")

plt.subplot(1,1,1)
if HSI_data == "wetland":
    plt.imshow(prediction, cmap=save_colormap_2)
else:
    plt.imshow(prediction)
plt.xticks([])
plt.yticks([])
plt.show()
if HSI_data == "wetland":
    plt.imsave(time_folder + r"\\prediction_result.png", prediction, cmap=save_colormap_2, dpi=300)
else: 
    plt.imsave(time_folder + r"\\prediction_result.png", prediction, dpi=300)

draw_result_visualization(time_folder, epoch_loss)
if model_type == "Transformer":
    store_result(time_folder, OA_val, AA_val, Kappa_val, CM_val, model_type, Transformer_mode, epoch, batch_size, patches, band_patches, learning_rate, weight_decay, gamma, sample_mode, sample_value)
elif model_type == "CNN":
    store_result(time_folder, OA_val, AA_val, Kappa_val, CM_val, model_type, CNN_mode, epoch, batch_size, patches, band_patches, learning_rate, weight_decay, gamma, sample_mode, sample_value)
savemat(time_folder + r"\\prediction_label.mat", {"prediction":prediction, "label":all_label})
