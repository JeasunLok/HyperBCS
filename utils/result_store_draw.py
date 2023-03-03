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
        f.write("Parameter settings:")
        f.write("model_type : " + model_type)
        f.write("model_mode : " + mode)
        f.write("epoch : " + str(epoch))
        f.write("batch_size : " + str(batch_size))
        f.write("patches : " + patches)
        if model_type == "Transformer":
            f.write("band_patches : " + band_patches)
        f.write("learning_rate : " + str(learning_rate))
        f.write("weight_decay : " + str(weight_decay))
        if model_type == "gamma":
            f.write("gamma : " + gamma)
        f.write("sample_mode : " + sample_mode)
        f.write("sample_mode : " + str(sample_value))
        f.write("Model result:")
        f.write("OA : {:.3f}\n".format(OA))
        f.write("AA : {:.3f}\n".format(AA))
        f.write("Kappa : {:.3f}\n".format(Kappa))
        f.write("Confusion Matrix :\n")
        f.write("{}".format(CM))
#-------------------------------------------------------------------------------