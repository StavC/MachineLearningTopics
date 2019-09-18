import numpy as np
import pandas as pd
import scipy
import  statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn


def main():

    sns.set()
    # doing a Linear Regression with one coef x1
    '''
    data=pd.read_csv('1.01. Simple linear regression.csv')
    #print(data)
    print(data.describe())
    y=data['GPA']
    x1=data['SAT']
    x=sm.add_constant(x1)
    results=sm.OLS(y,x).fit()
    print(results.summary())
    plt.scatter(x1,y)
    yhat=0.275+0.0017*x1 #regerssion equation from the summary
    fig=plt.plot(x1,yhat,lw=4,c='orange',label='regression line')
    plt.xlabel('SAT',fontsize=20)
    plt.ylabel('GPA',fontsize=20)
    plt.show()
    '''
    '''
    # doing a Linear Regression with two coef x1 and rand 1,2,3

    data=pd.read_csv('1.02. Multiple linear regression.csv')
    #GPA=b0+b1*SAT+b2*RAND(1,2,3)
    y=data['GPA']
    x1=data[['SAT','Rand 1,2,3']]
    x = sm.add_constant(x1)
    results = sm.OLS(y, x).fit()
    print(results.summary())
    ##### bad example lower  R-SQUARED
    '''
    '''
    data=pd.read_csv('real_estate_price_size_year.csv')
    print(data.describe())
    y=data['price']
    x1=data[['size','year']]
    x=sm.add_constant(x1)
    results=sm.OLS(y,x).fit()
    print(results.summary())
    '''

    raw_data=pd.read_csv('1.03. Dummies.csv')
    data=raw_data.copy()
    data['Attendance']=data['Attendance'].map({'Yes':1,'No':0})
    print(data.describe())
    ## Regerssion
    y=data['GPA']
    x1=data[['SAT','Attendance']]
    x=sm.add_constant(x1)
    results=sm.OLS(y,x).fit()
    print(results.summary())
    # Create one scatter plot which contains all observations
    # Use the series 'Attendance' as color, and choose a colour map of your choice
    # The colour map we've chosen is completely arbitrary
    plt.scatter(data['SAT'], data['GPA'], c=data['Attendance'], cmap='RdYlGn_r')
    # Define the two regression equations (one with a dummy = 1, the other with dummy = 0)
    # We have those above already, but for the sake of consistency, we will also include them here
    yhat_no = 0.6439 + 0.0014 * data['SAT']
    yhat_yes = 0.8665 + 0.0014 * data['SAT']
    # Original regression line
    yhat = 0.0017 * data['SAT'] + 0.275
    # Plot the two regression lines
    fig = plt.plot(data['SAT'], yhat_no, lw=2, c='#006837', label='regression line1')
    fig = plt.plot(data['SAT'], yhat_yes, lw=2, c='#a50026', label='regression line2')
    # Plot the original regression line
    fig = plt.plot(data['SAT'], yhat, lw=3, c='#4C72B0', label='regression line')

    plt.xlabel('SAT', fontsize=20)
    plt.ylabel('GPA', fontsize=20)
    plt.show()
    ##predicting
    data_frame=pd.DataFrame({'const':1,'SAT':[1700,1670],'Attendance':[0,1]})
    new_data1=data_frame[['const','SAT','Attendance']]
    print(new_data1)
    predictions=results.predict(new_data1)
    print(predictions)
    predictionsf=pd.DataFrame({'Predicstions':predictions})
    joined=new_data1.join(predictionsf)
    joined.rename(index={0:'Bob',1:'Alice'})
    print(joined)










if __name__ == '__main__':
    main()