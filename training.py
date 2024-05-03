import utils
from utils import Model_Trainer
from matplotlib import pyplot as plt

#Packages required:
#   numpy
#   torch/pytorch
#   matplotlib
#   opencv/cv2
#   tqdm

# This initializes the Model_Trainer class found in utils.py
# The parameters for the network structure that I found worked best are the
# defaults, but can be changed when initializing Model_Trainer.
# Borough is an integer that represents which borough to train on:
#       0 -> Manhattan
#       1 -> Brooklyn
#       2 -> Queens
#       3 -> Bronx
#       4 -> Staten Island
#       5 -> Don't use, but it represents no burglaries, so it is always used

model_trainer = Model_Trainer(borough=0)

# Begin training
model_trainer.train()

#Calculate in sample and out of sample accuracy
# acc_OS_tot, acc_OS_true, acc_OS_false, _, _ = model_trainer.eval_OS()
# acc_IS_tot, acc_IS_true, acc_IS_false, _, _ = model_trainer.eval_IS()
print('Out of sample accuracy: ', model_trainer.val_acc)
# print(acc_OS_tot)
# print(acc_IS_tot)

plt.figure()
plt.plot(model_trainer.losses)
plt.show()
print('finished training')
