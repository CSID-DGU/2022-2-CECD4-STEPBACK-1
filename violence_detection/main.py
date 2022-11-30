from M1 import *
from M2 import *
from M3 import * 
import sys
import glob
import os
import datetime
def convert_action_to_category(num):
    if num ==0 or num ==4 or num =='A01' or num =='A05': 
        return 0 #C011 =>0
    elif num ==1 or num==2 or num==3 or num ==5 or num==6 or num==7 or num =='A02' or num=='A03' or num=='A04' or num =='A06' or num=='A07' or num=='A08':
        return 1 #C012==>1
    elif num==8 or num ==9 or num==10 or num==11 or num==12 or num=='A17' or num =='A18' or num=='A19' or num=='A20' or num=='A21':
        return 2 #C021==>2
    elif num ==13 or num ==14  or num =='A22' or num =='A23':
        return 3 #C031==>3
    elif num ==16 or num==17 or num ==15 or num =='A24'or num =='A25' or num =='A26':
        return 4 #C032 ==>4
    elif num ==18 or num == 'A30':      
        return 5
    elif num ==19 or num =='A31':
        return 6     
    
    
                
def main(mp4_path , json_path , log_file='' ):
    now = datetime.datetime.now()
    input_list1=[]
    input_list2=[]
    labels =[]
    name = mp4_path.split('/')[-1]
    label = name.split('.')[0]
    label= label[5:8]
    name= name.split('.')[0]
    labels.append(convert_action_to_category(label))
    input_list1.append(convert_action_to_category(M1_test(mp4_path)))
    input_list2.append(convert_action_to_category(M2_test(json_path)))    
    M3(input_list1, input_list2 ,labels , name , now)    


if __name__ =="__main__":
    input1 = sys.argv[1] 
    input2 = sys.argv[2]
    input3 = sys.argv[3]
    if input1 == '-i':
        main(input2,input3)
    elif input1 =='-b':
        if os.path.isdir(input2) and os.path.isdir(input3):
            INPUT1 = glob.glob(input2+'/*.mp4')
            INPUT2 = glob.glob(input3+'/*.json')

            if len(INPUT1) == len(INPUT2):
                for i in range(len(INPUT1)):
                    main(INPUT1[i],INPUT2[i])
            else:
                print('두 디렉토리내 파일의 개수가 다릅니다.')

