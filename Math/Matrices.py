
import numpy as np
def main():


    ####Scalars
    s=5
    print(s)
    #### Vectors
    v=np.array([5,-2,4]) #row vector
    print(v)
    #### Matrices

    m=np.array([[5,12,6],[-3,0,14]])
    print(m)

    #### data types
    print(type(s))
    print(type(v))
    print(type(m))
    s_array=np.array(5)
    print(type(s_array))

    #### Data Shapes
    print(m.shape) # matrix 2 row 3 columns
    print(v.shape) #vector
    print(v)
    v=v.reshape(1,3) #1 row 3 col
    print(v)
    v=v.reshape(3,1)# 3 rows 1 col
    print(v)
    print(s_array.shape)# no size becasue only scalar





if __name__ == '__main__':
    main()