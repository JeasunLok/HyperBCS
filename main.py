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
from models.HyperMAC_1D import HyperMAC_1D
from models.HyperMAC_2D import HyperMAC_2D
from models.HyperMAC_3D import HyperMAC_3D
from models.HyperMAC_MultiScale_2D import HyperMAC_2D_MultiScale
from models.HyperMAC_MultiScale_2D_FCback import HyperMAC_2D_MultiScale_FCback
from models.HyperMAC_MultiScale_3D import HyperMAC_3D_MultiScale
from models.HyperMAC_MultiScale_3D_FCback import HyperMAC_3D_MultiScale_FCback
from models.ResNet_1D import ResNet_1D
from models.ResNet_2D import ResNet_2D
from models.ResNet_3D import ResNet_3D
from models.ResNet_MultiScale_2D import ResNet_MultiScale_2D
from models.ResNet_MultiScale_3D import ResNet_MultiScale_3D

from utils.data_processing import position_train_and_test_point,mirror_hsi,train_and_test_data,train_and_test_label
from utils.data_preparation import HSI_Dataset
from utils.metrics import output_metric
from utils.result_store_draw import draw_result_visualization,store_result
from utils.utils import load_wetland_data
from train import train_epoch,valid_epoch
from test import test_epoch

#-------------------------------------------------------------------------------
# setting the parameters
# model mode
mode = "train" # train or test
pretrained = False # pretrained or not
model_path = r"logs\2023-03-23-02-35-43-HyperMAC_MultiScale-3D-wetland2015\model_state_dict.pkl"

# model settings
model_type = "HyperMAC_MultiScale" # CNN_RNN or Transformer or HyperMAC or HyperMAC_MultiScale or ResNet or ResNet_MultiScale
Transformer_mode = "ViT" # if Transformer : ViT CAF
CNN_mode = "CNN_1D" # if CNN_RNN : MLP_4 CNN_1D CNN_2D CNN_3D CNN_3D_Classifer_1D RNN_1D
HyperMAC_mode = "1D" # if HyperMAC : 1D 2D 3D
ResNet_mode = "2D" # if ResNet : 1D 2D 3D
HyperMAC_MultiScale_mode = "3D" # if HyperMAC_MultiScale : 2D 3D
ResNet_MultiScale_mode = "3D" # if ResNet_MultiScale : 2D 3D
patches = 3 # if Transformer
band_patches = 3 # if Transformer

# training settings
gpu = 0
epoch = 100
#HyperMAC 3D =>40-50 HyperMAC 2D =>120 CNN_1D/CNN_3D/CNN_3D_Classifer_1D/RNN_1D =>500 others =>100-200
test_freq = 500
batch_size = 32
learning_rate = 5e-3
weight_decay = 0
gamma = 0.9

# data settings
sample_mode = "fixed" # fixed or percentage
sample_value = 200 # fixed => numble of samples(int)  percentage => percentage of samples(0-1) 
HSI_data = "wetland" # IndianPine or wetland
wetland_id = 2015 # if wetland
#-------------------------------------------------------------------------------

# make the run folder in logs
#-------------------------------------------------------------------------------
time_now = time.localtime()
if model_type == "Transformer":
    if HSI_data == "IndianPine":
        time_folder = r".\\logs\\" + time.strftime("%Y-%m-%d-%H-%M-%S", time_now) + "-" + model_type + "-" + Transformer_mode + "-" + HSI_data
    else:
        time_folder = r".\\logs\\" + time.strftime("%Y-%m-%d-%H-%M-%S", time_now) + "-" + model_type + "-" + Transformer_mode + "-" + HSI_data + str(wetland_id)

elif model_type == "CNN_RNN":
    if HSI_data == "IndianPine":
        time_folder = r".\\logs\\" + time.strftime("%Y-%m-%d-%H-%M-%S", time_now) + "-" + model_type + "-" + CNN_mode + "-" + HSI_data
    else:
        time_folder = r".\\logs\\" + time.strftime("%Y-%m-%d-%H-%M-%S", time_now) + "-" + model_type + "-" + CNN_mode + "-" + HSI_data + str(wetland_id)

elif model_type == "HyperMAC":
    if HSI_data == "IndianPine":
        time_folder = r".\\logs\\" + time.strftime("%Y-%m-%d-%H-%M-%S", time_now) + "-" + model_type + "-" + HyperMAC_mode + "-" + HSI_data
    else:
        time_folder = r".\\logs\\" + time.strftime("%Y-%m-%d-%H-%M-%S", time_now) + "-" + model_type + "-" + HyperMAC_mode + "-" + HSI_data + str(wetland_id)

elif model_type == "HyperMAC_MultiScale":
    if HSI_data == "IndianPine":
        time_folder = r".\\logs\\" + time.strftime("%Y-%m-%d-%H-%M-%S", time_now) + "-" + model_type + "-" + HyperMAC_MultiScale_mode + "-" + HSI_data
    else:
        time_folder = r".\\logs\\" + time.strftime("%Y-%m-%d-%H-%M-%S", time_now) + "-" + model_type + "-" + HyperMAC_MultiScale_mode + "-" + HSI_data + str(wetland_id)

elif model_type == "ResNet":
    if HSI_data == "IndianPine":
        time_folder = r".\\logs\\" + time.strftime("%Y-%m-%d-%H-%M-%S", time_now) + "-" + model_type + "-" + ResNet_mode + "-" + HSI_data
    else:
        time_folder = r".\\logs\\" + time.strftime("%Y-%m-%d-%H-%M-%S", time_now) + "-" + model_type + "-" + ResNet_mode + "-" + HSI_data + str(wetland_id)

elif model_type == "ResNet_MultiScale":
    if HSI_data == "IndianPine":
        time_folder = r".\\logs\\" + time.strftime("%Y-%m-%d-%H-%M-%S", time_now) + "-" + model_type + "-" + ResNet_MultiScale_mode + "-" + HSI_data
    else:
        time_folder = r".\\logs\\" + time.strftime("%Y-%m-%d-%H-%M-%S", time_now) + "-" + model_type + "-" + ResNet_MultiScale_mode + "-" + HSI_data + str(wetland_id)
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
    plt.imsave(time_folder + r"\\train_label.png", train_label, dpi=300)
    plt.imsave(time_folder + r"\\test_label.png", test_label, dpi=300)
    input = data['input'] #(145,145,200)
#-------------------------------------------------------------------------------

# wetland data
#-------------------------------------------------------------------------------
elif HSI_data == "wetland":
    # color settings
    colormap_mat = loadmat(r".\\data\\wetland_colormap.mat")
    colormap = colormap_mat["colormap_" + str(wetland_id)]
    colormap_1 = np.append("#FFFFFF", colormap)
    save_colormap_1 = mpl.colors.LinearSegmentedColormap.from_list('cmap', colormap_1.tolist(), 256)

    if model_type == "CNN_RNN" or model_type == "HyperMAC" or model_type == "HyperMAC_MultiScale" or model_type == "ResNet" or model_type == "ResNet_MultiScale":
        save_colormap_2 = save_colormap_1

    elif model_type == "Transformer":
        save_colormap_2 = mpl.colors.LinearSegmentedColormap.from_list('cmap', colormap.tolist(), 256)

    # data loading
    image_path = r".\\data\\" + str(wetland_id)[-2:] +"_image.mat"
    label_path = r".\\data\\" + str(wetland_id)[-2:] +"_label.mat"
    input, train_label, test_label = load_wetland_data(time_folder, image_path, label_path, wetland_id, sample_mode, sample_value, save_colormap_1)
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
            patch_size = 8
        )
        patches = 8

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

    # image and label should be mirrored
    mirror_image = mirror_hsi(height, width, band, input_normalize, patch=patches)
    mirror_train_label = mirror_hsi(height, width, 1, np.expand_dims(train_label, axis=2), patch=patches)
    mirror_test_label = mirror_hsi(height, width, 1, np.expand_dims(test_label, axis=2), patch=patches)

    mirror_train_label = mirror_train_label.reshape(mirror_image.shape[0],mirror_image.shape[1])
    mirror_test_label = mirror_test_label.reshape(mirror_image.shape[0],mirror_image.shape[1])

    train_dataset = HSI_Dataset(mirror_image, mirror_train_label, True, patches)
    train_loader = Data.DataLoader(train_dataset, batch_size, shuffle = True)

    test_dataset = HSI_Dataset(mirror_image, mirror_test_label, True, patches)
    test_loader = Data.DataLoader(test_dataset, batch_size, shuffle = True)

    true_dataset = HSI_Dataset(mirror_image, mirror_test_label, False, patches)
    true_loader = Data.DataLoader(true_dataset, batch_size, shuffle = False)

    total_pos_true = true_dataset.indices
#-------------------------------------------------------------------------------

# HyperMAC models
#-------------------------------------------------------------------------------
elif model_type == "HyperMAC":
    if HyperMAC_mode == "1D":
        model = HyperMAC_1D(
            input_channels = 1,
            num_classes = num_classes + 1
        )
        patches = 1

    if HyperMAC_mode == "2D":
        model = HyperMAC_2D(
            input_channels = band,
            num_classes = num_classes + 1
        )
        patches = 8
    
    if HyperMAC_mode == "3D":
        model = HyperMAC_3D(
            input_channels = band,
            num_classes = num_classes + 1
        )
        patches = 8

    # image and label should be mirrored
    mirror_image = mirror_hsi(height, width, band, input_normalize, patch=patches)
    mirror_train_label = mirror_hsi(height, width, 1, np.expand_dims(train_label, axis=2), patch=patches)
    mirror_test_label = mirror_hsi(height, width, 1, np.expand_dims(test_label, axis=2), patch=patches)

    mirror_train_label = mirror_train_label.reshape(mirror_image.shape[0],mirror_image.shape[1])
    mirror_test_label = mirror_test_label.reshape(mirror_image.shape[0],mirror_image.shape[1])

    train_dataset = HSI_Dataset(mirror_image, mirror_train_label, True, patches)
    train_loader = Data.DataLoader(train_dataset, batch_size, shuffle = True)

    test_dataset = HSI_Dataset(mirror_image, mirror_test_label, True, patches)
    test_loader = Data.DataLoader(test_dataset, batch_size, shuffle = True)

    true_dataset = HSI_Dataset(mirror_image, mirror_test_label, False, patches)
    true_loader = Data.DataLoader(true_dataset, batch_size, shuffle = False)

    total_pos_true = true_dataset.indices
#-------------------------------------------------------------------------------

# HyperMAC_MultiScale models
#-------------------------------------------------------------------------------
elif model_type == "HyperMAC_MultiScale":
    if HyperMAC_MultiScale_mode == "2D":
        model = HyperMAC_2D_MultiScale(
            input_channels = band,
            num_classes = num_classes + 1
        )
        patches = 8
    
    if HyperMAC_MultiScale_mode == "3D":
        model = HyperMAC_3D_MultiScale(
            input_channels = band,
            num_classes = num_classes + 1
        )
        patches = 8

    # image and label should be mirrored
    mirror_image = mirror_hsi(height, width, band, input_normalize, patch=patches)
    mirror_train_label = mirror_hsi(height, width, 1, np.expand_dims(train_label, axis=2), patch=patches)
    mirror_test_label = mirror_hsi(height, width, 1, np.expand_dims(test_label, axis=2), patch=patches)

    mirror_train_label = mirror_train_label.reshape(mirror_image.shape[0],mirror_image.shape[1])
    mirror_test_label = mirror_test_label.reshape(mirror_image.shape[0],mirror_image.shape[1])

    train_dataset = HSI_Dataset(mirror_image, mirror_train_label, True, patches)
    train_loader = Data.DataLoader(train_dataset, batch_size, shuffle = True)

    test_dataset = HSI_Dataset(mirror_image, mirror_test_label, True, patches)
    test_loader = Data.DataLoader(test_dataset, batch_size, shuffle = True)

    true_dataset = HSI_Dataset(mirror_image, mirror_test_label, False, patches)
    true_loader = Data.DataLoader(true_dataset, batch_size, shuffle = False)

    total_pos_true = true_dataset.indices
#-------------------------------------------------------------------------------

# ResNet models
#-------------------------------------------------------------------------------
elif model_type == "ResNet":
    if ResNet_mode == "1D":
        model = ResNet_1D(
            input_channels = 1,
            num_classes = num_classes + 1
        )
        patches = 1

    if ResNet_mode == "2D":
        model = ResNet_2D(
            input_channels = band,
            num_classes = num_classes + 1
        )
        patches = 8
    
    if ResNet_mode == "3D":
        model = ResNet_3D(
            input_channels = band,
            num_classes = num_classes + 1
        )
        patches = 8

    # image and label should be mirrored
    mirror_image = mirror_hsi(height, width, band, input_normalize, patch=patches)
    mirror_train_label = mirror_hsi(height, width, 1, np.expand_dims(train_label, axis=2), patch=patches)
    mirror_test_label = mirror_hsi(height, width, 1, np.expand_dims(test_label, axis=2), patch=patches)

    mirror_train_label = mirror_train_label.reshape(mirror_image.shape[0],mirror_image.shape[1])
    mirror_test_label = mirror_test_label.reshape(mirror_image.shape[0],mirror_image.shape[1])

    train_dataset = HSI_Dataset(mirror_image, mirror_train_label, True, patches)
    train_loader = Data.DataLoader(train_dataset, batch_size, shuffle = True)

    test_dataset = HSI_Dataset(mirror_image, mirror_test_label, True, patches)
    test_loader = Data.DataLoader(test_dataset, batch_size, shuffle = True)

    true_dataset = HSI_Dataset(mirror_image, mirror_test_label, False, patches)
    true_loader = Data.DataLoader(true_dataset, batch_size, shuffle = False)

    total_pos_true = true_dataset.indices
#-------------------------------------------------------------------------------

# ResNet_MultiScale models
#-------------------------------------------------------------------------------
elif model_type == "ResNet_MultiScale":
    if ResNet_MultiScale_mode == "2D":
        model = ResNet_MultiScale_2D(
            input_channels = band,
            num_classes = num_classes + 1
        )
        patches = 8
    
    if ResNet_MultiScale_mode == "3D":
        model = ResNet_MultiScale_3D(
            input_channels = band,
            num_classes = num_classes + 1
        )
        patches = 8

    # image and label should be mirrored
    mirror_image = mirror_hsi(height, width, band, input_normalize, patch=patches)
    mirror_train_label = mirror_hsi(height, width, 1, np.expand_dims(train_label, axis=2), patch=patches)
    mirror_test_label = mirror_hsi(height, width, 1, np.expand_dims(test_label, axis=2), patch=patches)

    mirror_train_label = mirror_train_label.reshape(mirror_image.shape[0],mirror_image.shape[1])
    mirror_test_label = mirror_test_label.reshape(mirror_image.shape[0],mirror_image.shape[1])

    train_dataset = HSI_Dataset(mirror_image, mirror_train_label, True, patches)
    train_loader = Data.DataLoader(train_dataset, batch_size, shuffle = True)

    test_dataset = HSI_Dataset(mirror_image, mirror_test_label, True, patches)
    test_loader = Data.DataLoader(test_dataset, batch_size, shuffle = True)

    true_dataset = HSI_Dataset(mirror_image, mirror_test_label, False, patches)
    true_loader = Data.DataLoader(true_dataset, batch_size, shuffle = False)

    total_pos_true = true_dataset.indices
#-------------------------------------------------------------------------------

# model settings
#-------------------------------------------------------------------------------
model = model.cuda()
# criterion
criterion = nn.CrossEntropyLoss().cuda()
# optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
# scheduler
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=epoch//10, gamma=gamma)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epoch//10, eta_min=5e-4)

#-------------------------------------------------------------------------------
if mode == "train":
    # if pretrained
    if pretrained:
        model.load_state_dict(torch.load(model_path))
        print("load model path : " + model_path)
    # train
    print("===============================================================================")
    print("start training")
    tic = time.time()
    epoch_result = np.zeros([3, epoch])
    for e in range(epoch): 
        model.train()
        train_acc, train_loss, label_t, prediction_t = train_epoch(model, train_loader, criterion, optimizer, e, epoch)
        scheduler.step()
        OA_train, AA_train, Kappa_train, CA_train, CM_train = output_metric(label_t, prediction_t) 
        print("Epoch: {:03d} | train_loss: {:.4f} | train_acc: {:.4f}".format(e+1, train_loss, train_acc))
        epoch_result[0][e], epoch_result[1][e], epoch_result[2][e] = e+1, train_loss, train_acc

        if ((e+1) % test_freq == 0) | (e == epoch - 1):
            print("===============================================================================")
            print("start validating")      
            model.eval()
            label_v, prediction_v = valid_epoch(model, test_loader, criterion, optimizer)
            OA_val, AA_val, Kappa_val, CA_val, CM_val = output_metric(label_v, prediction_v)
            if (e != epoch -1):
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

elif mode == "test":
    model.load_state_dict(torch.load(model_path))
    print("load model path : " + model_path)
    print("===============================================================================")

if mode == "train":
    draw_result_visualization(time_folder, epoch_result)
    if model_type == "Transformer":
        store_result(time_folder, OA_val, AA_val, Kappa_val, CM_val, model_type, Transformer_mode, epoch, batch_size, patches, band_patches, learning_rate, weight_decay, gamma, sample_mode, sample_value)
    elif model_type == "CNN_RNN":
        store_result(time_folder, OA_val, AA_val, Kappa_val, CM_val, model_type, CNN_mode, epoch, batch_size, patches, band_patches, learning_rate, weight_decay, gamma, sample_mode, sample_value)
    elif model_type == "HyperMAC":
        store_result(time_folder, OA_val, AA_val, Kappa_val, CM_val, model_type, HyperMAC_mode, epoch, batch_size, patches, band_patches, learning_rate, weight_decay, gamma, sample_mode, sample_value)
    elif model_type == "HyperMAC_MultiScale":
        store_result(time_folder, OA_val, AA_val, Kappa_val, CM_val, model_type, HyperMAC_MultiScale_mode, epoch, batch_size, patches, band_patches, learning_rate, weight_decay, gamma, sample_mode, sample_value)
    elif model_type == "ResNet":
        store_result(time_folder, OA_val, AA_val, Kappa_val, CM_val, model_type, ResNet_mode, epoch, batch_size, patches, band_patches, learning_rate, weight_decay, gamma, sample_mode, sample_value)
    elif model_type == "ResNet_MultiScale":
        store_result(time_folder, OA_val, AA_val, Kappa_val, CM_val, model_type, ResNet_MultiScale_mode, epoch, batch_size, patches, band_patches, learning_rate, weight_decay, gamma, sample_mode, sample_value)
    # save model and its parameters 
    torch.save(model, time_folder + r"\\model.pkl")
    torch.save(model.state_dict(), time_folder + r"\\model_state_dict.pkl")

print("start testing")
model.eval()

# output classification maps
padding = patches // 2
pre_u = test_epoch(model, true_loader)
prediction = np.zeros((height, width), dtype=float)
prediction_temp = np.zeros((height+2*padding, width+2*padding), dtype=float)
for i in range(total_pos_true.shape[0]):
    if model_type == "Transformer":
        prediction[total_pos_true[i,0], total_pos_true[i,1]] = pre_u[i] + 1
    elif model_type == "CNN_RNN" or model_type == "HyperMAC" or model_type == "HyperMAC_MultiScale" or model_type == "ResNet" or model_type == "ResNet_MultiScale":
        prediction_temp[total_pos_true[i,0], total_pos_true[i,1]] = pre_u[i]

if model_type == "CNN_RNN" or model_type =="HyperMAC" or model_type == "HyperMAC_MultiScale" or model_type == "ResNet" or model_type == "ResNet_MultiScale":
    for i in range(height):
        for j in range(width):
            prediction[i,j] = prediction_temp[padding+i,padding+j]

print("end testing")
print("===============================================================================")

# result show
plt.figure()
plt.subplot(1,1,1)
if HSI_data == "wetland":
    plt.imshow(prediction, cmap=save_colormap_2)
else:
    plt.imshow(prediction)
plt.xticks([])
plt.yticks([])
plt.show()

# image and result save
if HSI_data == "wetland":
    plt.imsave(time_folder + r"\\image.png", input_normalize[:,:,[14,7,2]], dpi=300)
    plt.imsave(time_folder + r"\\prediction.png", prediction, cmap=save_colormap_2, dpi=300)
else: 
    plt.imsave(time_folder + r"\\prediction.png", prediction, dpi=300)

# save the predict image
savemat(time_folder + r"\\prediction_label.mat", {"prediction":prediction, "label":all_label})
