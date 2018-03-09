import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense,Activation,Dropout
from keras import optimizers

from keras.datasets import mnist
from keras.utils import np_utils
import keras.callbacks

from sklearn.decomposition import PCA   
import os
from keras.callbacks import TensorBoard
import glob
import cv2

def build_model(input_node_num):
        
        
    model = Sequential()
    ###   code here  #####
    model.add(Dense(40,input_dim=5))
    model.add(Activation('relu'))
    #model.add(Dropout(0.2))
    model.add(Dense(40))
    model.add(Activation('relu'))
    #model.add(Dropout(0.2))
    model.add(Dense(4))
    model.add(Activation('softmax'))
    ###   code here  #####
    model.summary()
    return model
if __name__ == "__main__":

    

    
    ############### train  number #####################
    n_train = 0
    train_list = glob.glob('train/*')
    for id,dir in enumerate(train_list):
        n_train += len(glob.glob(dir+'/*'))
    print 'n_train',n_train,'\n'
    x_train = np.zeros((n_train,30*30*3), dtype=float)
    y_train = np.zeros((n_train,4), dtype=int)
    
    
    
    ################ test  number ###############
    n_test = 0
    test_list = glob.glob('test/*')
    for id,dir in enumerate(test_list):
        n_test += len(glob.glob(dir+'/*'))
    print 'n_test',n_test,'\n'
    x_test = np.zeros((n_test,30*30*3), dtype=float)
    y_test = np.zeros((n_test,4), dtype=int)
    
    
    
    
    ################ read train image ###############
    print 'read train image'
    train_index = 0
    for id,dir in enumerate(sorted(train_list)):
        print id,dir
        for index,file in enumerate(glob.glob(dir+'/*')):
            x_train[train_index,:] = cv2.imread(file).reshape((30*30*3))/255.0
            y_train[train_index,:] = np_utils.to_categorical(id, 4)
            train_index += 1
    
    
    
    ################ read test image ###############
    print '\nread test image'
    test_index = 0
    for id,dir in enumerate(sorted(test_list)):
        print id,dir
        for index,file in enumerate(glob.glob(dir+'/*')):
            x_test[test_index,:] = cv2.imread(file).reshape((30*30*3))/255.0
            y_test[test_index,:] = np_utils.to_categorical(id, 4)
            test_index += 1
    
    
    
    input_node_num = 5
    model = build_model(input_node_num)
    pca=PCA(n_components=input_node_num) 
    x_train_new=pca.fit_transform(x_train)  
    
    sgd = optimizers.SGD(lr=0.005, clipnorm=1.)
    model.compile(loss='categorical_crossentropy',optimizer='sgd',metrics=['accuracy'])
    

    model.fit(x_train_new,y_train,batch_size=20,epochs=100,callbacks=[TensorBoard(log_dir='./log_dir')])
    
    score = model.evaluate(x_train_new,y_train)
    print '\n\nTrain Acc:', score[1]
    
    x_test_new=pca.transform(x_test)  
    score = model.evaluate(x_test_new,y_test)
    print '\n\nTest Acc:', score[1]
    
    
    
    
    
