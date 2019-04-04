#to train:  
#this will create log dirctory, and log files, including 2 model  
python cnn.py --lr=0.001  
python cnn.py --lr=0.005  
  
#to interpolate:  
#this will create loss and accuracy file in log directory  
#you should run train before interpolation  
python interpolation.py  
  
#to plot:  
#this will create interpolation.png in current directory  
python plot.py  
