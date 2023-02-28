import numpy as np

# the position of train and test samples
# 定位训练和测试样本
#-------------------------------------------------------------------------------
def position_train_and_test_point(train_data, test_data, label_data, num_classes):
    number_train = [] # count the number of train samples 计数训练样本数量
    position_train = {} # record the position of train samples 记录训练样本位置
    number_test = [] # count the number of test samples 计数测试样本数量
    position_test = {} # record the position of test samples 记录测试样本位置
    number_label = [] # count the number of true samples 计数总样本数量
    position_label = {} # record the position of true samples 记录总样本位置
    #-------------------------for train data------------------------------------
    for i in range(num_classes):
        each_class = []
        # find the position of every class  
        each_class = np.argwhere(train_data==(i+1))

        # count the column of samples' position => the number of samples
        number_train.append(each_class.shape[0]) 
        position_train[i] = each_class # 

    # initialize the train data(return)
    total_position_train = position_train[0]
    # concatenate the train data of every class 
    # => every samples' position are recorded in order according to the class
    for i in range(1, num_classes):
        total_position_train = np.r_[total_position_train, position_train[i]] #(695,2)

    # convert the data to type int
    total_position_train = total_position_train.astype(int)

    #--------------------------for test data------------------------------------
    for i in range(num_classes):
        each_class = []
        each_class = np.argwhere(test_data==(i+1))
        number_test.append(each_class.shape[0])
        position_test[i] = each_class

    total_position_test = position_test[0]
    for i in range(1, num_classes):
        total_position_test = np.r_[total_position_test, position_test[i]] #(9671,2)
    total_position_test = total_position_test.astype(int)

    #--------------------------for true data------------------------------------
    for i in range(num_classes+1):
        each_class = []
        each_class = np.argwhere(label_data==i)
        number_label.append(each_class.shape[0])
        position_label[i] = each_class

    total_position_label = position_label[0]
    for i in range(1, num_classes+1):
        total_position_label = np.r_[total_position_label, position_label[i]]
    total_position_label = total_position_label.astype(int)

    return total_position_train, total_position_test, total_position_label, number_train, number_test, number_label
#-------------------------------------------------------------------------------

# expand the border by mirror
# 边界拓展：镜像
# the pixels' value closed to the border must be valid when divided into patches
#-------------------------------------------------------------------------------
def mirror_hsi(height, width, band, input_normalize, patch=5):
    padding = patch//2
    mirror_hsi = np.zeros((height+2*padding,width+2*padding,band),dtype=float)
    # center zone don't need to pad by mirror 中心区域
    mirror_hsi[padding:(padding+height),padding:(padding+width),:] = input_normalize
    # left mirror 左边镜像
    for i in range(padding):
        mirror_hsi[padding:(height+padding),i,:] = input_normalize[:,padding-i-1,:]
    # right mirror 右边镜像
    for i in range(padding):
        mirror_hsi[padding:(height+padding),width+padding+i,:] = input_normalize[:,width-1-i,:]
    # up mirror 上边镜像
    for i in range(padding):
        mirror_hsi[i,:,:] = mirror_hsi[padding*2-i-1,:,:]
    # bottom mirror 下边镜像
    for i in range(padding):
        mirror_hsi[height+padding+i,:,:] = mirror_hsi[height+padding-1-i,:,:]

    print("===============================================================================")
    print("patch is : {}".format(patch))
    print("mirror_image shape : [{0},{1},{2}]".format(mirror_hsi.shape[0],mirror_hsi.shape[1],mirror_hsi.shape[2]))
    print("===============================================================================")
    return mirror_hsi
#-------------------------------------------------------------------------------

# get the image value of 2D-patch
# 获取二维patch的图像数据
#-------------------------------------------------------------------------------
def gain_neighborhood_pixel(mirror_image, point, i, patch=5):
    # x,y is the coordinate of the sample i
    x = point[i,0]
    y = point[i,1]
    temp_image = mirror_image[x:(x+patch),y:(y+patch),:]
    return temp_image
#-------------------------------------------------------------------------------

# get the image value of 3D-patch
# 获取三维patch的图像数据
#-------------------------------------------------------------------------------
def gain_neighborhood_band(x_train, band, band_patch, patch=5):
    nn = band_patch // 2
    pp = (patch*patch) // 2

    # x_train : [numbers of train samples, patch, patch, band] => [numbers of train samples, patch^2, band]
    x_train_reshape = x_train.reshape(x_train.shape[0], patch*patch, band)

    # x_train_band : x_train with band-patch which means 2D => 3D
    x_train_band = np.zeros((x_train.shape[0], patch*patch*band_patch, band),dtype=float)

    # center zone 中心区域
    x_train_band[:,nn*patch*patch:(nn+1)*patch*patch,:] = x_train_reshape

    # band patch is in a sequence so it only needs to mirror in 1D
    # left mirror 左边镜像
    for i in range(nn):
        if pp > 0: # equal to pp==0
            x_train_band[:,i*patch*patch:(i+1)*patch*patch,:i+1] = x_train_reshape[:,:,band-i-1:]
            x_train_band[:,i*patch*patch:(i+1)*patch*patch,i+1:] = x_train_reshape[:,:,:band-i-1]
        else:
            x_train_band[:,i:(i+1),:(nn-i)] = x_train_reshape[:,0:1,(band-nn+i):]
            x_train_band[:,i:(i+1),(nn-i):] = x_train_reshape[:,0:1,:(band-nn+i)]

    # right mirror 右边镜像
    for i in range(nn):
        if pp > 0: # equal to pp==0
            x_train_band[:,(nn+i+1)*patch*patch:(nn+i+2)*patch*patch,:band-i-1] = x_train_reshape[:,:,i+1:]
            x_train_band[:,(nn+i+1)*patch*patch:(nn+i+2)*patch*patch,band-i-1:] = x_train_reshape[:,:,:i+1]
        else:
            x_train_band[:,(nn+1+i):(nn+2+i),(band-i-1):] = x_train_reshape[:,0:1,:(i+1)]
            x_train_band[:,(nn+1+i):(nn+2+i),:(band-i-1)] = x_train_reshape[:,0:1,(i+1):]
    return x_train_band
#-------------------------------------------------------------------------------

# to summarize the train data and test data by using gain neighbor pixel and band
# 汇总训练数据和测试数据
#-------------------------------------------------------------------------------
def train_and_test_data(mirror_image, band, train_point, test_point, label_point, patch=5, band_patch=3):
    x_train = np.zeros((train_point.shape[0], patch, patch, band), dtype=float)
    x_test = np.zeros((test_point.shape[0], patch, patch, band), dtype=float)
    x_label = np.zeros((label_point.shape[0], patch, patch, band), dtype=float)

    # 1. get 2D patch
    for i in range(train_point.shape[0]):
        x_train[i,:,:,:] = gain_neighborhood_pixel(mirror_image, train_point, i, patch)
    for j in range(test_point.shape[0]):
        x_test[j,:,:,:] = gain_neighborhood_pixel(mirror_image, test_point, j, patch)
    for k in range(label_point.shape[0]):
        x_label[k,:,:,:] = gain_neighborhood_pixel(mirror_image, label_point, k, patch)

    print("x_train shape = {}, type = {}".format(x_train.shape,x_train.dtype))
    print("x_test  shape = {}, type = {}".format(x_test.shape,x_test.dtype))
    print("x_true  shape = {}, type = {}".format(x_label.shape,x_test.dtype))
    print("===============================================================================")
    
    # 2. using 2D patch to get 3D patch
    x_train_band = gain_neighborhood_band(x_train, band, band_patch, patch)
    x_test_band = gain_neighborhood_band(x_test, band, band_patch, patch)
    x_label_band = gain_neighborhood_band(x_label, band, band_patch, patch)
    print("x_train_band shape = {}, type = {}".format(x_train_band.shape,x_train_band.dtype))
    print("x_test_band  shape = {}, type = {}".format(x_test_band.shape,x_test_band.dtype))
    print("x_true_band  shape = {}, type = {}".format(x_label_band.shape,x_label_band.dtype))
    print("===============================================================================")
    return x_train_band, x_test_band, x_label_band
#-------------------------------------------------------------------------------

# create the label of the train data and test data
# 创建标签y_train, y_test
#-------------------------------------------------------------------------------
def train_and_test_label(number_train, number_test, number_label, num_classes):
    y_train = []
    y_test = []
    y_label = []

    # The lable is in order which means we can using the number of each class's samples to get the label
    for i in range(num_classes):
        for j in range(number_train[i]):
            y_train.append(i)
        for k in range(number_test[i]):
            y_test.append(i)
    for i in range(num_classes+1):
        for j in range(number_label[i]):
            y_label.append(i)

    y_train = np.array(y_train)
    y_test = np.array(y_test)
    y_label = np.array(y_label)

    print("y_train: shape = {} ,type = {}".format(y_train.shape,y_train.dtype))
    print("y_test: shape = {} ,type = {}".format(y_test.shape,y_test.dtype))
    print("y_true: shape = {} ,type = {}".format(y_label.shape,y_label.dtype))
    print("===============================================================================")
    return y_train, y_test, y_label
#-------------------------------------------------------------------------------