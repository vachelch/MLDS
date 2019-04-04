# import numpy as np
import matplotlib.pyplot as plt
paras = [2172, 51666, 138330, 303018, 550570, 668602, 1208138,1671114, 2262922, 5626378]

loss = [0.805379, 0.456469, 0.159477, 0.052376, 0.100771, 0.156388, 0.003622, 0.128376, 0.016067, 0.049119]
acc = [0.675000, 0.812500, 0.925000, 0.987500, 0.975000, 0.975000, 1.000000, 0.950000, 1.000000, 0.987500]

test_loss = [1.093192, 0.852735, 1.066756, 1.457985, 1.721015, 1.320283, 1.287716, 1.334546, 1.409798, 1.690462]
test_acc = [0.603000, 0.711000, 0.735000, 0.745000, 0.732000, 0.780000, 0.774000, 0.773000, 0.772000, 0.749000]

plt.figure()
plt.scatter(paras, loss, label='train_loss')
plt.scatter(paras, test_loss, label='test_loss')
plt.xlabel('number of parameters')
plt.ylabel('loss')
plt.title('model loss')
plt.legend()
plt.savefig('model_loss.png')
plt.show()


plt.figure()
plt.scatter(paras, acc, label='train_acc')
plt.scatter(paras, test_acc, label='test_acc')
plt.xlabel('number of parameters')
plt.ylabel('accuracy')
plt.title('model accuracy')
plt.legend()
plt.savefig('model_acc.png')
plt.show()