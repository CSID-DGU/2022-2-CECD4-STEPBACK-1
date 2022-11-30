# coding=utf8
#from models import c3d_model
from ast import While
from sre_constants import CATEGORY

from torch import true_divide
from tensorflow.keras.optimizers import SGD
import tensorflow as tf
import numpy as np
import cv2
import os
import json
from PIL import Image,ImageDraw,ImageFont
from datetime import datetime
from glob import glob
from tqdm import tqdm
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from keras.layers import Dense,Dropout,Conv3D,Input,MaxPool3D,Flatten,Activation
from keras.regularizers import l2
from keras.models import Model
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

video_name = 'paper1'

def c3d_model():
    input_shape = (112,112,16,3)
    weight_decay = 0.005
    nb_classes = 20

    inputs = Input(input_shape)
    x = Conv3D(64,(3,3,3),strides=(1,1,1),padding='same',
               activation='relu',kernel_regularizer=l2(weight_decay))(inputs)
    x = MaxPool3D((2,2,1),strides=(2,2,1),padding='same')(x)

    x = Conv3D(128,(3,3,3),strides=(1,1,1),padding='same',
               activation='relu',kernel_regularizer=l2(weight_decay))(x)
    x = MaxPool3D((2,2,2),strides=(2,2,2),padding='same')(x)

    x = Conv3D(128,(3,3,3),strides=(1,1,1),padding='same',
               activation='relu',kernel_regularizer=l2(weight_decay))(x)
    x = MaxPool3D((2,2,2),strides=(2,2,2),padding='same')(x)

    x = Conv3D(256,(3,3,3),strides=(1,1,1),padding='same',
               activation='relu',kernel_regularizer=l2(weight_decay))(x)
    x = MaxPool3D((2,2,2),strides=(2,2,2),padding='same')(x)

    x = Conv3D(256, (3, 3, 3), strides=(1, 1, 1), padding='same',
               activation='relu',kernel_regularizer=l2(weight_decay))(x)
    x = MaxPool3D((2, 2, 2), strides=(2, 2, 2), padding='same')(x)

    x = Flatten()(x)
    x = Dense(2048,activation='relu',kernel_regularizer=l2(weight_decay))(x)
    x = Dropout(0.5)(x)
    x = Dense(2048,activation='relu',kernel_regularizer=l2(weight_decay))(x)
    x = Dropout(0.5)(x)
    x = Dense(nb_classes,kernel_regularizer=l2(weight_decay))(x)
    x = Activation('softmax')(x)

    model = Model(inputs, x)
    return model

#def M1_test(path):
def M1_test(path):
    #path = './test/mp4/C041/C041_A30_SY32_P07_S06_02DAS.mp4'
    now = datetime.now()
    times = str(now)
    test_date=str(datetime.today().month) +'.'+ str(datetime.today().day)  
    print(path)
    video_name=path.split('/')[-1]
    
    
    fm=open('./input_data/index.txt', 'r')
    main_names = fm.readlines()
    CATEGORY = video_name[:-4]
    if not os.path.exists('./test_log'):
        os.mkdir('./test_log')
    if not os.path.exists('./test_log/'+CATEGORY):
        os.mkdir('./test_log/'+CATEGORY)
    
    file_name = video_name.split('.')[0]      
    fw =open('./test_log/'+CATEGORY+'/'+file_name+'.txt', 'w')
    # init model
    model = c3d_model()
    #lr = 0.005
    lr = 0.9
    sgd = SGD(learning_rate=lr, momentum=0.9, nesterov=True)
    #sgd = SGD(learning_rate=lr, momentum=2, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    model = tf.keras.models.load_model('./input_data/epoch10_temp_weights_c3d.h5')

    cap = cv2.VideoCapture(path)
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(3))
    height = int(cap.get(4))
    
    clip = []
    main_count_list = [0 for i in range(len(main_names))]
    scene=0
    fw.write(str(fps)+'\n')
    fw.write(str(frame_count)+'\n')
    start = time.time()
    for i in tqdm(range(frame_count)):
        duration = cap.get(cv2.CAP_PROP_POS_MSEC) # frame_count/fps
        minutes = int((duration / 1000) / 60)
        seconds = (duration / 1000) % 60

        print('Time (M.S) = ' + str(minutes) + ':' + str(seconds))

        ret, frame = cap.read()
        if ret:
            tmp = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            clip.append(cv2.resize(tmp, (171, 128)))
            if len(clip) == 16:
                inputs = np.array(clip).astype(np.float32)
                inputs = np.expand_dims(inputs, axis=0)
                inputs[..., 0] -= 99.9
                inputs[..., 1] -= 92.1
                inputs[..., 2] -= 82.6
                inputs[..., 0] /= 65.8
                inputs[..., 1] /= 62.3
                inputs[..., 2] /= 60.3
                inputs = inputs[:,:,8:120,30:142,:]
                inputs = np.transpose(inputs, (0, 2, 3, 1, 4))

                
                pred_main = model.predict(inputs)
                main_label = np.argmax(pred_main[0])        
                main_count_list[main_label]=main_count_list[main_label]+1
                # if main_names[main_label].split(' ')[1].strip() in ('A02', 'A03', 'A04', 'A05', 'A06', 'A07', 'A08', 'A19', 'A22', 'A23', 'A24', 'A25', 'A26'):
                fw.write(str(i) +" %d" % (pred_main[0][main_label]*100)+'\n')
                clip.pop(0)
        
    end = datetime.now()    
    end_time = str(end)
    ftw = open('./test_log/'+CATEGORY+'/'+file_name+'_M1_'+end_time[0:19]+'_total.txt', 'w')
    ftw.write(video_name+'\n')
    ftw.write('영상 '+str(frame_count-15)+' 프레임 중 ')
    main_mode_label = np.argmax(main_count_list)    
    ftw.write(main_names[main_mode_label].split(' ')[-1].strip()+" 검출 "+str(main_count_list[main_mode_label])+" 프레임 ")\
    
    main_frame_prod=main_count_list[main_mode_label]/(frame_count-15)*100
    
    return_value =main_names[main_mode_label].split(' ')[-1].strip()
    ftw.write(str(int(main_frame_prod))+'%\n')
    for corr_main_label in range(len(main_names)):
        if video_name==main_names[corr_main_label].split(' ')[-1].strip()!=main_names[main_mode_label].split(' ')[-1].strip():
            main_frame_prod=main_count_list[corr_main_label]/(frame_count-15)*100
            ftw.write('\t\t\t\t\t\t\t'+main_names[corr_main_label].split(' ')[-1].strip()+" 검출 "+str(main_count_list[corr_main_label])+" 프레임 "+str(int(main_frame_prod))+'%\n')
    return return_value
   
M1_test('test/mp4/'+ video_name +'.mp4') 

cap = cv2.VideoCapture('test/mp4/'+ video_name +'.mp4')
fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
w = round(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
f = open('test_log/'+ video_name +'/'+ video_name +'.txt', 'r')

fourcc = cv2.VideoWriter_fourcc(*'DIVX')
out = cv2.VideoWriter('test_log/'+ video_name +'/'+ video_name + ' output.mp4', fourcc, fps/30, (w, h))
lines = f.readlines()
text_list = []
for i in range(0, 14):
    text_list.append(['0', '0'])

for i in range(2, len(lines)):
    line = lines[i].strip()
    tmp = line.split(' ')
    text_list.append(tmp)

idx = 0
while True:
    ret, frame = cap.read()
    str = "Frame Num: " + text_list[idx][0]
    str2 = "Accuracy: " + text_list[idx][1] + "%"
    cv2.putText(frame, str, (10, 50), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255))
    cv2.putText(frame, str2, (10, 100), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255))
    cv2.imshow('frame', frame)
    out.write(frame)

    if cv2.waitKey(1) == 27:
        break
    idx += 1