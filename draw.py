import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = "Times New Roman"
LR_2D_5e_3 = np.loadtxt(r"logs\Learning Rate\2023-03-24-22-30-32-HyperMAC_MultiScale-2D-wetland2015\epoch.txt", dtype=float, delimiter=",")
LR_3D_5e_3 = np.loadtxt(r"logs\Learning Rate\2023-03-25-00-35-45-HyperMAC_MultiScale-3D-wetland2015\epoch.txt", dtype=float, delimiter=",")
result_2D_5e_3 = 0.962
result_3D_5e_3 = 0.974

LR_2D_5e_4 = np.loadtxt(r"logs\Learning Rate\2023-03-24-22-36-17-HyperMAC_MultiScale-2D-wetland2015\epoch.txt", dtype=float, delimiter=",")
LR_3D_5e_4 = np.loadtxt(r"logs\Learning Rate\2023-03-25-03-13-53-HyperMAC_MultiScale-3D-wetland2015\epoch.txt", dtype=float, delimiter=",")
result_2D_5e_4 = 0.973
result_3D_5e_4 = 0.954

LR_2D_5e_5 = np.loadtxt(r"logs\Learning Rate\2023-03-24-23-28-33-HyperMAC_MultiScale-2D-wetland2015\epoch.txt", dtype=float, delimiter=",")
LR_3D_5e_2 = np.loadtxt(r"logs\Learning Rate\2023-03-25-01-25-32-HyperMAC_MultiScale-3D-wetland2015\epoch.txt", dtype=float, delimiter=",")
result_2D_5e_5 = 0.930
result_3D_5e_2 = 0.941

LR_2D_1e_3 = np.loadtxt(r"logs\Learning Rate\2023-03-24-23-52-33-HyperMAC_MultiScale-2D-wetland2015\epoch.txt", dtype=float, delimiter=",")
LR_3D_1e_3 = np.loadtxt(r"logs\Learning Rate\2023-03-25-00-10-56-HyperMAC_MultiScale-3D-wetland2015\epoch.txt", dtype=float, delimiter=",")
result_2D_1e_3 = 0.960
result_3D_1e_3 = 0.942

LR_2D_1e_4 = np.loadtxt(r"logs\Learning Rate\2023-03-24-23-46-41-HyperMAC_MultiScale-2D-wetland2015\epoch.txt", dtype=float, delimiter=",")
LR_3D_1e_4 = np.loadtxt(r"logs\Learning Rate\2023-03-25-03-38-24-HyperMAC_MultiScale-3D-wetland2015\epoch.txt", dtype=float, delimiter=",")
result_2D_1e_4 = 0.974
result_3D_1e_4 = 0.918

LR_2D_1e_5 = np.loadtxt(r"logs\Learning Rate\2023-03-24-23-41-54-HyperMAC_MultiScale-2D-wetland2015\epoch.txt", dtype=float, delimiter=",")
LR_3D_1e_2 = np.loadtxt(r"logs\Learning Rate\2023-03-25-01-52-54-HyperMAC_MultiScale-3D-wetland2015\epoch.txt", dtype=float, delimiter=",")
result_2D_1e_5 = 0.953
result_3D_1e_2 = 0.983

epoch = LR_2D_5e_3[0]
result_2D = np.array([result_2D_1e_5, result_2D_5e_5, result_2D_1e_4, result_2D_5e_4, result_2D_1e_3, result_2D_5e_3])
LR_2D = ["0.00001", "0.00005", "0.0001", "0.0005", "0.001", "0.005"]
result_3D = np.array([result_3D_1e_4, result_3D_5e_4, result_3D_1e_3, result_3D_5e_3, result_3D_1e_2, result_3D_5e_2])
LR_3D = ["0.0001", "0.0005", "0.001", "0.005", "0.01", "0.05"]

figure, ax12 = plt.subplots()
# plt.plot(epoch[0:200:4], LR_2D_5e_3[1][0:200:4], '--', marker="o", markersize=3, label="lr=0.005", color="#333C42", linewidth=2)
plt.plot(epoch[0:200:4], LR_2D_5e_3[1][0:200:4], '-', marker="o", markersize=3, label="lr=0.005", color="#A80326", linewidth=2)
# plt.plot(epoch[0:200:4], LR_2D_5e_4[1][0:200:4], '-', marker="v", markersize=3, label="lr=0.0005", color="#316658", linewidth=2)
plt.plot(epoch[0:200:4], LR_2D_5e_4[1][0:200:4], '-', marker="v", markersize=3, label="lr=0.0005", color="#EC5D3B", linewidth=2)
# plt.plot(epoch[0:200:4], LR_2D_5e_5[1][0:200:4], '--', marker="^", markersize=3, label="lr=0.00005", color="#5EA69C", linewidth=2)
plt.plot(epoch[0:200:4], LR_2D_5e_5[1][0:200:4], '-', marker="^", markersize=3, label="lr=0.00005", color="#FDB96B", linewidth=2)
# plt.plot(epoch[0:200:4], LR_2D_1e_3[1][0:200:4], '-', marker="s", markersize=3, label="lr=0.001", color="#C2CFA2", linewidth=2)
plt.plot(epoch[0:200:4], LR_2D_1e_3[1][0:200:4], '-', marker="s", markersize=3, label="lr=0.001", color="#CAE8F2", linewidth=2)
# plt.plot(epoch[0:200:4], LR_2D_1e_4[1][0:200:4], '--', marker="8", markersize=3, label="lr=0.0001", color="#A4799E", linewidth=2)
plt.plot(epoch[0:200:4], LR_2D_1e_4[1][0:200:4], '-', marker="8", markersize=3, label="lr=0.0001", color="#72AACF", linewidth=2)
# plt.plot(epoch[0:200:4], LR_2D_1e_5[1][0:200:4], '-', marker="p", markersize=3, label="lr=0.00001", color="#706690", linewidth=2)
plt.plot(epoch[0:200:4], LR_2D_1e_5[1][0:200:4], '-', marker="X", markersize=3, label="lr=0.00001", color="#3951A2", linewidth=2)
ax12.grid(axis='y', linestyle='-.', alpha=0.1)# 背景网格
ax12.set_xticks(np.arange(0, 201, 25))
ax12.set_yticks(np.arange(0, 2.51, 0.25))
plt.xlabel("Epoch",  fontsize=12)
plt.ylabel("Trainning loss", fontsize=12)
ax12.legend(fontsize=10, frameon=False, ncol=1, loc='upper right')# 图例
plt.savefig(r".\\images\\Hyper2MAC_2D Epoch,Learning rate and Loss(0-200).png", dpi=300)
plt.show()

figure, ax13 = plt.subplots()
# plt.plot(epoch[0:200:4], LR_2D_5e_3[1][0:200:4], '--', marker="o", markersize=3, label="lr=0.005", color="#333C42", linewidth=2)
plt.plot(epoch[0:150:3], LR_3D_5e_2[1][0:150:3], '-', marker="o", markersize=3, label="lr=0.005", color="#A80326", linewidth=2)
# plt.plot(epoch[0:200:4], LR_2D_5e_4[1][0:200:4], '-', marker="v", markersize=3, label="lr=0.0005", color="#316658", linewidth=2)
plt.plot(epoch[0:150:3], LR_3D_5e_3[1][0:150:3], '-', marker="v", markersize=3, label="lr=0.0005", color="#EC5D3B", linewidth=2)
# plt.plot(epoch[0:200:4], LR_2D_5e_5[1][0:200:4], '--', marker="^", markersize=3, label="lr=0.00005", color="#5EA69C", linewidth=2)
plt.plot(epoch[0:150:3], LR_3D_5e_4[1][0:150:3], '-', marker="^", markersize=3, label="lr=0.00005", color="#FDB96B", linewidth=2)
# plt.plot(epoch[0:200:4], LR_2D_1e_3[1][0:200:4], '-', marker="s", markersize=3, label="lr=0.001", color="#C2CFA2", linewidth=2)
plt.plot(epoch[0:150:3], LR_3D_1e_2[1][0:150:3], '-', marker="s", markersize=3, label="lr=0.001", color="#CAE8F2", linewidth=2)
# plt.plot(epoch[0:200:4], LR_2D_1e_4[1][0:200:4], '--', marker="8", markersize=3, label="lr=0.0001", color="#A4799E", linewidth=2)
plt.plot(epoch[0:150:3], LR_3D_1e_3[1][0:150:3], '-', marker="8", markersize=3, label="lr=0.0001", color="#72AACF", linewidth=2)
# plt.plot(epoch[0:200:4], LR_2D_1e_5[1][0:200:4], '-', marker="p", markersize=3, label="lr=0.00001", color="#706690", linewidth=2)
plt.plot(epoch[0:150:3], LR_3D_1e_4[1][0:150:3], '-', marker="X", markersize=3, label="lr=0.00001", color="#3951A2", linewidth=2)
ax13.grid(axis='y', linestyle='-.', alpha=0.1)# 背景网格
ax13.set_xticks(np.arange(0, 151, 15))
ax13.set_yticks(np.arange(0, 2.26, 0.25))
plt.xlabel("Epoch",  fontsize=12)
plt.ylabel("Trainning loss", fontsize=12)
ax13.legend(fontsize=10, frameon=False, ncol=1, loc='upper right')# 图例
plt.savefig(r".\\images\\Hyper2MAC_3D Epoch,Learning rate and Loss(0-150).png", dpi=300)
plt.show()

figure, ax14 = plt.subplots()
# plt.plot(epoch[0:200:4], LR_2D_5e_3[1][0:200:4], '--', marker="o", markersize=3, label="lr=0.005", color="#333C42", linewidth=2)
plt.plot(epoch[60:150:3], LR_3D_5e_2[1][60:150:3], '-', marker="o", markersize=3, label="lr=0.005", color="#A80326", linewidth=2)
# plt.plot(epoch[0:200:4], LR_2D_5e_4[1][0:200:4], '-', marker="v", markersize=3, label="lr=0.0005", color="#316658", linewidth=2)
plt.plot(epoch[60:150:3], LR_3D_5e_3[1][60:150:3], '-', marker="v", markersize=3, label="lr=0.0005", color="#EC5D3B", linewidth=2)
# plt.plot(epoch[0:150:3], LR_2D_5e_5[1][0:200:4], '--', marker="^", markersize=3, label="lr=0.00005", color="#5EA69C", linewidth=2)
plt.plot(epoch[60:150:3], LR_3D_5e_4[1][60:150:3], '-', marker="^", markersize=3, label="lr=0.00005", color="#FDB96B", linewidth=2)
# plt.plot(epoch[0:150:3], LR_2D_1e_3[1][0:200:4], '-', marker="s", markersize=3, label="lr=0.001", color="#C2CFA2", linewidth=2)
plt.plot(epoch[60:150:3], LR_3D_1e_2[1][60:150:3], '-', marker="s", markersize=3, label="lr=0.001", color="#CAE8F2", linewidth=2)
# plt.plot(epoch[0:150:3], LR_2D_1e_4[1][0:200:4], '--', marker="8", markersize=3, label="lr=0.0001", color="#A4799E", linewidth=2)
plt.plot(epoch[60:150:3], LR_3D_1e_3[1][60:150:3], '-', marker="8", markersize=3, label="lr=0.0001", color="#72AACF", linewidth=2)
# plt.plot(epoch[0:200:4], LR_2D_1e_5[1][0:200:4], '-', marker="p", markersize=3, label="lr=0.00001", color="#706690", linewidth=2)
plt.plot(epoch[60:150:3], LR_3D_1e_4[1][60:150:3], '-', marker="X", markersize=3, label="lr=0.00001", color="#3951A2", linewidth=2)
ax14.grid(axis='y', linestyle='-.', alpha=0.1)# 背景网格
ax14.set_xticks(np.arange(60, 151, 10))
ax14.set_yticks(np.arange(0, 0.201, 0.025))
plt.xlabel("Epoch",  fontsize=12)
plt.ylabel("Trainning loss", fontsize=12)
ax14.legend(fontsize=10, frameon=False, ncol=1, loc='upper right')# 图例
plt.savefig(r".\\images\\Hyper2MAC_3D Epoch,Learning rate and Loss(60-150).png", dpi=300)
plt.show()

figure, ax11 = plt.subplots()
# plt.plot(epoch[0:200:4], LR_2D_5e_3[1][0:200:4], '--', marker="o", markersize=3, label="lr=0.005", color="#333C42", linewidth=2)
plt.plot(epoch[80:200:4], LR_2D_5e_3[1][80:200:4], '-', marker="o", markersize=3, label="lr=0.005", color="#A80326", linewidth=2)
# plt.plot(epoch[0:200:4], LR_2D_5e_4[1][0:200:4], '-', marker="v", markersize=3, label="lr=0.0005", color="#316658", linewidth=2)
plt.plot(epoch[80:200:4], LR_2D_5e_4[1][80:200:4], '-', marker="v", markersize=3, label="lr=0.0005", color="#EC5D3B", linewidth=2)
# plt.plot(epoch[0:200:4], LR_2D_5e_5[1][0:200:4], '--', marker="^", markersize=3, label="lr=0.00005", color="#5EA69C", linewidth=2)
plt.plot(epoch[80:200:4], LR_2D_5e_5[1][80:200:4], '-', marker="^", markersize=3, label="lr=0.00005", color="#FDB96B", linewidth=2)
# plt.plot(epoch[0:200:4], LR_2D_1e_3[1][0:200:4], '-', marker="s", markersize=3, label="lr=0.001", color="#C2CFA2", linewidth=2)
plt.plot(epoch[80:200:4], LR_2D_1e_3[1][80:200:4], '-', marker="s", markersize=3, label="lr=0.001", color="#CAE8F2", linewidth=2)
# plt.plot(epoch[0:200:4], LR_2D_1e_4[1][0:200:4], '--', marker="8", markersize=3, label="lr=0.0001", color="#A4799E", linewidth=2)
plt.plot(epoch[80:200:4], LR_2D_1e_4[1][80:200:4], '-', marker="8", markersize=3, label="lr=0.0001", color="#72AACF", linewidth=2)
# plt.plot(epoch[0:200:4], LR_2D_1e_5[1][0:200:4], '-', marker="p", markersize=3, label="lr=0.00001", color="#706690", linewidth=2)
plt.plot(epoch[80:200:4], LR_2D_1e_5[1][80:200:4], '-', marker="X", markersize=3, label="lr=0.00001", color="#3951A2", linewidth=2)
ax11.grid(axis='y', linestyle='-.', alpha=0.1)# 背景网格
ax11.set_xticks(np.arange(80, 201, 20))
ax11.set_yticks(np.arange(0, 0.271, 0.03))
plt.xlabel("Epoch",  fontsize=12)
plt.ylabel("Trainning loss", fontsize=12)
ax11.legend(fontsize=10, frameon=False, ncol=1, loc='upper right')# 图例
plt.savefig(r".\\images\\Hyper2MAC_2D Epoch,Learning rate and Loss(80-200).png", dpi=300)
plt.show()

result_2D_training = np.array([LR_2D_1e_5[2][-1], LR_2D_5e_5[2][-1], LR_2D_1e_4[2][-1], LR_2D_5e_4[2][-1], LR_2D_1e_3[2][-1], LR_2D_5e_3[2][-1]])
LR_2D_xaxis = np.arange(0,6)
figure, ax21 = plt.subplots()
ax21.grid(axis='y', linestyle='-.', alpha=0.1)# 背景网格
width = 0.3
plt.bar(LR_2D_xaxis-width/2, result_2D*100, width=width, label="Testing accuracy", color="#E71343")
plt.bar(LR_2D_xaxis+width/2, result_2D_training, width=width, label="Training accuracy", color="#FFD460")
ax21.set_ylim(92,100.9)
plt.yticks(np.arange(92,100.1,1))
plt.xticks(LR_2D_xaxis, LR_2D)
plt.xlabel("Learning rate",  fontsize=12)
plt.ylabel("Accuracy", fontsize=12)
ax21.legend(fontsize=10, frameon=False, ncol=2, loc='upper right')# 图例
plt.savefig(r".\\images\\Hyper2MAC_2D Learning rate and Accuracy.png", dpi=300)
plt.show()

result_3D_training = np.array([LR_3D_1e_4[2][-1], LR_3D_5e_4[2][-1], LR_3D_1e_3[2][-1], LR_3D_5e_3[2][-1], LR_3D_1e_2[2][-1], LR_3D_5e_2[2][-1]])
LR_3D_xaxis = np.arange(0,6)
figure, ax22 = plt.subplots()
ax22.grid(axis='y', linestyle='-.', alpha=0.1)# 背景网格
width = 0.3
plt.bar(LR_3D_xaxis-width/2, result_3D*100, width=width, label="Testing accuracy", color="#E71343")
plt.bar(LR_3D_xaxis+width/2, result_3D_training, width=width, label="Training accuracy", color="#FFD460")
ax22.set_ylim(91,101)
plt.yticks(np.arange(91,100.1,1))
plt.xticks(LR_3D_xaxis, LR_3D)
plt.xlabel("Learning rate",  fontsize=12)
plt.ylabel("Accuracy", fontsize=12)
ax22.legend(fontsize=10, frameon=False, ncol=2, loc='upper right')# 图例
plt.savefig(r".\\images\\Hyper2MAC_3D Learning rate and Accuracy.png", dpi=300)
plt.show()


Best_3D_8 = LR_3D_1e_2
Best_3D_8_result = result_3D_1e_2
Best_2D_8 = LR_2D_1e_4
Best_2D_8_result = result_2D_1e_4

Patch_3D_16 = np.loadtxt(r"logs\Patches\2023-03-25-04-27-50-HyperMAC_MultiScale-3D-wetland2015\epoch.txt", dtype=float, delimiter=",")
Patch_3D_16_result = 0.988
Patch_2D_16 = np.loadtxt(r"logs\Patches\2023-03-25-05-15-50-HyperMAC_MultiScale-2D-wetland2015\epoch.txt", dtype=float, delimiter=",")
Patch_2D_16_result = 0.985

Patch_xaxis = np.arange(0,2)
figure, ax31 = plt.subplots()
ax31.grid(axis='y', linestyle='-.', alpha=0.1)# 背景网格
width = 0.3
plt.bar(Patch_xaxis-width/2, np.array([Best_2D_8_result*100, Best_3D_8_result*100]), width=width, label="patches=8", color="#FEEC37")
plt.bar(Patch_xaxis+width/2, np.array([Patch_2D_16_result*100, Patch_3D_16_result*100]), width=width, label="patches=16", color="#2C38A3")
ax31.set_ylim(97,99.01)
plt.yticks(np.arange(97,99.01,0.25))
plt.xticks(Patch_xaxis, ["2D-HyperMBS 2D", "3D-HyperBCS"])
plt.xlabel("Network",  fontsize=12)
plt.ylabel("Accuracy", fontsize=12)
ax31.legend(fontsize=10, frameon=False, ncol=2, loc='upper right')# 图例
plt.savefig(r".\\images\\Hyper2MAC Patches Accuracy.png", dpi=300)
plt.show()