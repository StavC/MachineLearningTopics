import numpy as np
from sklearn.model_selection import train_test_split

def main():

    a=np.arange(1,101)
    b=np.arange(501,601)
    a_train,a_test=train_test_split(a)
    print(a_train.shape,a_test.shape)
    a_train,a_test,b_train,b_test=train_test_split(a,b,test_size=0.2,random_state=42)
    print(a_train.shape,a_test.shape)






if __name__ == '__main__':
    main()