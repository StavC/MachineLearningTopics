import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import f_regression
from sklearn.preprocessing import  StandardScaler


def main():

    sns.set()
    '''
    data=pd.read_csv('1.01. Simple linear regression.csv')
    x=data['SAT']
    y=data['GPA']
    reg=LinearRegression()
    print(x.shape)
    x_matrix=x.values.reshape(-1,1)
    print(x_matrix.shape)
    print(reg.fit(x_matrix,y))
    print(reg.score(x_matrix,y))
    print(reg.coef_)
    print(reg.intercept_)## getting the const
    new_data=pd.DataFrame(data=[1740,1760],columns=['SAT'])
    print(new_data)
    new_data['Predicted_GPA']=reg.predict(new_data)
    print(new_data)

    plt.scatter(x,y)
    yhat=reg.coef_*x_matrix+reg.intercept_
    #yhat=0.0017*x+0.275
    fig = plt.plot(x, yhat, lw=4, c='orange', label='regression line')
    plt.xlabel('SAT', fontsize=20)
    plt.ylabel('GPA', fontsize=20)
    plt.show()
    '''

    data=pd.read_csv('1.02. Multiple linear regression.csv')
    x=data[['SAT','Rand 1,2,3']]
    y=data['GPA']
    reg=LinearRegression()
    reg.fit(x,y)
    print(reg.coef_)
    print(reg.intercept_)
    print(f"R^2: {reg.score(x,y)}")
    r2=reg.score(x,y)
    n=x.shape[0]
    p=x.shape[1]
    print(n)
    adjusted_r2=1-(1-r2)*(n-1)/(n-p-1)
    print(f"adjusted r2{adjusted_r2}")
    print(f_regression(x,y))
    p_values=f_regression(x,y)[1]
    p_values=p_values.round(3)
    print(p_values)
    ## we can remove rand beacuse 0.676
    ################# SUMMARY TABLE
    reg_summary=pd.DataFrame(data=x.columns.values,columns=['Features'])
    reg_summary['Coefficients']=reg.coef_
    reg_summary['p-values']=p_values.round(3)
    reg_summary['R2']=reg.score(x,y)
    reg_summary['Adjusted R2']=1-(1-r2)*(n-1)/(n-p-1)
    print(reg_summary)

    ###############  Standardization
    scaler=StandardScaler()
    scaler.fit(x)
    x_scaled=scaler.transform(x)
    #print(x_scaled)
    reg=LinearRegression()
    reg.fit(x_scaled,y)
    reg_summary=pd.DataFrame([['Bias'],['SAT'],['Rand 1,2,3']],columns=['Features'])
    reg_summary['Weights']=reg.intercept_,reg.coef_[0],reg.coef_[1]
    # the bigger the weight the bigger the impact
    print(reg_summary)
    new_data=pd.DataFrame(data=[[1700,2],[1800,1]],columns=['SAT','Rand 1,2,3'])
    new_data_scaled=scaler.transform(new_data)
    print(reg.predict(new_data_scaled))






if __name__ == '__main__':
    main()