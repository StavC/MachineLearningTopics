
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

    #### Creating a tensor

    m1=np.array([[5,12,6],[-3,0,14]])
    m2=np.array([[9,8,7],[1,3,-5]])
    t=np.array([m1,m2])
    print(t)
    print(t.shape)

    #### Transposing Matrices
    print(m1)
    m11=m1.transpose()
    print(m11)

    #### dot Product
    x=np.array([2,8,-4])
    y=np.array([1,-7,3])
    dot=np.dot(x,y)
    print(dot)

    m3=np.array([[2,-1],[8,0],[3,0]])
    print(np.dot(m1,m3))





if __name__ == '__main__':
    main()