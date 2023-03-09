import numpy as np
import matplotlib.pyplot as plt

#-------------------------------------------------------------------------------
# the function to draw the result : to do
def draw_result_visualization(folder, epoch_loss):
    # the change of loss
    plt.plot(epoch_loss[:][0], epoch_loss[:][1])
    plt.title("the change of the loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.savefig(folder + r"\\loss_change.png")
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
# the function to store the accuracy of the val and test
def store_result(foloder, OA, AA, Kappa, CM, model_type, mode, epoch, batch_size, patches, band_patches, learning_rate, weight_decay, gamma, sample_mode, sample_value):
    with open(foloder + r"\\accuracy.txt", 'w', encoding="utf-8") as f:
        f.write("Parameter settings:" + "\n")
        f.write("model_type : " + model_type + "\n")
        f.write("model_mode : " + mode + "\n")
        f.write("epoch : " + str(epoch) + "\n")
        f.write("batch_size : " + str(batch_size) + "\n")
        f.write("patches : " + str(patches) + "\n")
        if model_type == "Transformer":
            f.write("band_patches : " + str(band_patches) + "\n")
        f.write("learning_rate : " + str(learning_rate) + "\n")
        f.write("weight_decay : " + str(weight_decay) + "\n")
        if model_type == "gamma":
            f.write("gamma : " + str(gamma) + "\n")
        f.write("sample_mode : " + sample_mode + "\n")
        f.write("sample_value : " + str(sample_value) + "\n")
        f.write("Model result:" + "\n")
        f.write("OA : {:.3f}\n".format(OA))
        f.write("AA : {:.3f}\n".format(AA))
        f.write("Kappa : {:.3f}\n".format(Kappa))
        f.write("Confusion Matrix :\n")
        f.write("{}".format(CM))
#-------------------------------------------------------------------------------