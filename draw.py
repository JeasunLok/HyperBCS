import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = "Times New Roman"
LR_2D_5e_3 = np.loadtxt(r"logs\Learning Rate\2023-03-24-22-30-32-HyperMAC_MultiScale-2D-wetland2015\epoch.txt", dtype=float, delimiter=",")
result_2D_5e_3 = 0.962
LR_2D_5e_4 = np.loadtxt(r"logs\Learning Rate\2023-03-24-22-36-17-HyperMAC_MultiScale-2D-wetland2015\epoch.txt", dtype=float, delimiter=",")
result_2D_5e_4 = 0.973
LR_2D_5e_5 = np.loadtxt(r"logs\Learning Rate\2023-03-24-23-28-33-HyperMAC_MultiScale-2D-wetland2015\epoch.txt", dtype=float, delimiter=",")
result_2D_5e_5 = 0.930
LR_2D_1e_3 = np.loadtxt(r"logs\Learning Rate\2023-03-24-23-52-33-HyperMAC_MultiScale-2D-wetland2015\epoch.txt", dtype=float, delimiter=",")
result_2D_1e_3 = 0.960
LR_2D_1e_4 = np.loadtxt(r"logs\Learning Rate\2023-03-24-23-46-41-HyperMAC_MultiScale-2D-wetland2015\epoch.txt", dtype=float, delimiter=",")
result_2D_1e_4 = 0.974
LR_2D_1e_5 = np.loadtxt(r"logs\Learning Rate\2023-03-24-23-41-54-HyperMAC_MultiScale-2D-wetland2015\epoch.txt", dtype=float, delimiter=",")
result_2D_1e_5 = 0.953
epoch = LR_2D_5e_3[0]

figure, ax1 = plt.subplots()
# plt.plot(epoch[0:200:4], LR_2D_5e_3[1][0:200:4], '--', marker="o", markersize=3, label="lr=0.005", color="#333C42", linewidth=2)
plt.plot(epoch[0:200:4], LR_2D_5e_3[1][0:200:4], '--', marker="o", markersize=3, label="lr=0.005", color="#A80326", linewidth=2)
# plt.plot(epoch[0:200:4], LR_2D_5e_4[1][0:200:4], '-', marker="v", markersize=3, label="lr=0.0005", color="#316658", linewidth=2)
plt.plot(epoch[0:200:4], LR_2D_5e_4[1][0:200:4], '-', marker="v", markersize=3, label="lr=0.0005", color="#EC5D3B", linewidth=2)
# plt.plot(epoch[0:200:4], LR_2D_5e_5[1][0:200:4], '--', marker="^", markersize=3, label="lr=0.00005", color="#5EA69C", linewidth=2)
plt.plot(epoch[0:200:4], LR_2D_5e_5[1][0:200:4], '--', marker="^", markersize=3, label="lr=0.00005", color="#FDB96B", linewidth=2)
# plt.plot(epoch[0:200:4], LR_2D_1e_3[1][0:200:4], '-', marker="s", markersize=3, label="lr=0.001", color="#C2CFA2", linewidth=2)
plt.plot(epoch[0:200:4], LR_2D_1e_3[1][0:200:4], '-', marker="s", markersize=3, label="lr=0.001", color="#CAE8F2", linewidth=2)
# plt.plot(epoch[0:200:4], LR_2D_1e_4[1][0:200:4], '--', marker="8", markersize=3, label="lr=0.0001", color="#A4799E", linewidth=2)
plt.plot(epoch[0:200:4], LR_2D_1e_4[1][0:200:4], '--', marker="8", markersize=3, label="lr=0.0001", color="#72AACF", linewidth=2)
# plt.plot(epoch[0:200:4], LR_2D_1e_5[1][0:200:4], '-', marker="p", markersize=3, label="lr=0.00001", color="#706690", linewidth=2)
plt.plot(epoch[0:200:4], LR_2D_1e_5[1][0:200:4], '-', marker="X", markersize=3, label="lr=0.00001", color="#3951A2", linewidth=2)
ax1.grid(axis='y', linestyle='-.', alpha=0.1)# 背景网格
ax1.set_xticks(np.arange(0, 201, 25))
ax1.set_yticks(np.arange(0, 2.51, 0.25))
plt.xlabel("Epoch",  fontsize=12)
plt.ylabel("Trainning loss", fontsize=12)
ax1.legend(fontsize=10, frameon=False, ncol=1, loc='upper right')# 图例
plt.show()

figure, ax11 = plt.subplots()
# plt.plot(epoch[0:200:4], LR_2D_5e_3[1][0:200:4], '--', marker="o", markersize=3, label="lr=0.005", color="#333C42", linewidth=2)
plt.plot(epoch[80:200:4], LR_2D_5e_3[1][80:200:4], '--', marker="o", markersize=3, label="lr=0.005", color="#A80326", linewidth=2)
# plt.plot(epoch[0:200:4], LR_2D_5e_4[1][0:200:4], '-', marker="v", markersize=3, label="lr=0.0005", color="#316658", linewidth=2)
plt.plot(epoch[80:200:4], LR_2D_5e_4[1][80:200:4], '-', marker="v", markersize=3, label="lr=0.0005", color="#EC5D3B", linewidth=2)
# plt.plot(epoch[0:200:4], LR_2D_5e_5[1][0:200:4], '--', marker="^", markersize=3, label="lr=0.00005", color="#5EA69C", linewidth=2)
plt.plot(epoch[80:200:4], LR_2D_5e_5[1][80:200:4], '--', marker="^", markersize=3, label="lr=0.00005", color="#FDB96B", linewidth=2)
# plt.plot(epoch[0:200:4], LR_2D_1e_3[1][0:200:4], '-', marker="s", markersize=3, label="lr=0.001", color="#C2CFA2", linewidth=2)
plt.plot(epoch[80:200:4], LR_2D_1e_3[1][80:200:4], '-', marker="s", markersize=3, label="lr=0.001", color="#CAE8F2", linewidth=2)
# plt.plot(epoch[0:200:4], LR_2D_1e_4[1][0:200:4], '--', marker="8", markersize=3, label="lr=0.0001", color="#A4799E", linewidth=2)
plt.plot(epoch[80:200:4], LR_2D_1e_4[1][80:200:4], '--', marker="8", markersize=3, label="lr=0.0001", color="#72AACF", linewidth=2)
# plt.plot(epoch[0:200:4], LR_2D_1e_5[1][0:200:4], '-', marker="p", markersize=3, label="lr=0.00001", color="#706690", linewidth=2)
plt.plot(epoch[80:200:4], LR_2D_1e_5[1][80:200:4], '-', marker="X", markersize=3, label="lr=0.00001", color="#3951A2", linewidth=2)
ax11.grid(axis='y', linestyle='-.', alpha=0.1)# 背景网格
ax11.set_xticks(np.arange(80, 201, 20))
ax11.set_yticks(np.arange(0, 0.271, 0.03))
plt.xlabel("Epoch",  fontsize=12)
plt.ylabel("Trainning loss", fontsize=12)
ax11.legend(fontsize=10, frameon=False, ncol=1, loc='upper right')# 图例
plt.show()



# figure, ax2 = plt.subplots()
# plt.plot(epoch,LR_2D_5e_3[2])
# plt.plot(epoch,LR_2D_5e_4[2])
# plt.plot(epoch,LR_2D_5e_5[2])
# plt.plot(epoch,LR_2D_1e_3[2])
# plt.plot(epoch,LR_2D_1e_4[2])
# plt.plot(epoch,LR_2D_1e_5[2])
# plt.xlabel("Epoch")
# plt.ylabel("Training accuracy")
# plt.show()

# figure, ax3 = plt.subplots()
# LR_2D = np.array([result_2D_1e_5, result_2D_5e_5, result_2D_1e_4, result_2D_5e_4, result_2D_1e_3, result_2D_5e_3])
# plt.plot(np.linspace(1,6,6), LR_2D)
# plt.xlabel("Learning rate")
# plt.ylabel("Testing accuracy")
# plt.show()