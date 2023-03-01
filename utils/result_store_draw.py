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
def store_result(foloder, OA, AA, Kappa, CM):
    with open(foloder + r"\\accuracy.txt", 'w', encoding="utf-8") as f:
        f.write("OA : {:.3f}\n".format(OA))
        f.write("AA : {:.3f}\n".format(AA))
        f.write("Kappa : {:.3f}\n".format(Kappa))
        f.write("Confusion Matrix :\n")
        f.write("{}".format(CM))
#-------------------------------------------------------------------------------