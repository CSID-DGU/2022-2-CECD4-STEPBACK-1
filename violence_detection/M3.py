import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score
import datetime
import joblib


def calculate_F1(matrix, f):
    precision = []
    recall= []
    for i in range(len(matrix)):
        answer = matrix[i][i]
        col_sum=matrix.sum(axis= 0)[i]  #precision
        row_sum=matrix.sum(axis=1)[i]   #recall
        precision.append(answer/col_sum)
        recall.append(answer/row_sum)
    precision_value = sum(precision) / len(precision)
    print('precision_value: ', precision_value)
    recall_value = sum(recall) / len(recall)
    print('recall_value: ', recall_value)
    F1 = (2*precision_value*recall_value) / (precision_value + recall_value)
    f.write('Precision: '+str(precision_value)+'\n')
    f.write('Recall: '+str(recall_value)+'\n')
    f.write('F1-Score: '+str(F1)+'\n\n')
    #print(F1)
    return F1  
def num_to_category(input):
    if input ==0:
        return 'C011'
    elif input ==1:
        return 'C012'
    elif input ==2:
        return 'C021'
    elif input ==3:
        return 'C031'
    elif input ==4:
        return 'C032'
    elif input ==5:
        return 'C041'
    elif input ==6:
        return 'C042'
def M3(input1, input2 , label , names ,now):
    now2 = datetime.datetime.now()
    time = str(now2)
    CATEGORY = names
    CATEGORY = CATEGORY[0:4]
    file_name = names

    f =open('./test_log/'+CATEGORY+'/'+file_name+'_M3_'+time[0:19]+'_.txt', 'w')
    
    f.write('Start time : '+str(now)+'\n\n')
    
    X_test = [[0 for col in range(2)] for row in range(len(input1))]
    
    for i in range(len(input1)):
        X_test[i][0] = input1[i]
        X_test[i][1] = input2[i]
    X_test = np.array(X_test)    
    y_test = np.array(label)
    sc =  joblib.load('./input_data/Standard_Scaler.pkl')
    clf = joblib.load('./input_data/SVM_Model.pkl')
    X_test_std = sc.transform(X_test)
    y_pred = clf.predict(X_test_std)
    
    count =0
    for i in range(len(input1)):
        print('정답: ',num_to_category(y_test[i]))
        answer = num_to_category(y_test[i])
        print('예측값: ',num_to_category(y_pred[i]))
        pred = num_to_category(y_pred[i])
        if y_test[i] == y_pred[i]:
            f.write('Target file: \n'+names+',\nPrediction : '+pred+'  Correct: '+answer+'\n\n')
            count +=1
        else:
            f.write('Target file: \n'+names+',\nPrediction : '+pred+'  Correct: '+answer+'\n\n')
            print(names[i]+'는 틀렸습니다.')    
    #f.write('Correct answer(s):'+str(count)+' / '+str(len(names))+'\n')
    #f.write('Accuracy: '+str(count/len(names))+'\n')        
    #F1 = calculate_F1(confusion_matrix(y_test, y_pred),f)
    end = datetime.datetime.now()
    f.write('Finish time: '+str(end))        
            
        
    
