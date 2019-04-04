#to train:  
#this will create log dirctory, and log files, including 5 model  
python cnn.py --lr=0.001  
python cnn.py --lr=0.002  
python cnn.py --lr=0.003  
python cnn.py --lr=0.004  
python cnn.py --lr=0.005  
  
#to get sensitivity:  
#this will create loss and accuracy file in log directory  
#you should run train before to get sensitivity  
python sensitivity.py  
  
#to plot:  
#this will create loss_sensi.png, acc_sensi.png in current directory  
python plot.py  
