import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
#sns.set()

#Apply a fix to the statsmodels library
from scipy import stats
stats.chisqprob = lambda chisq, df: stats.chi2.sf(chisq, df)


def main():

    raw_data=pd.read_csv('Example-bank-data.csv')
    #print(raw_data)
    data=raw_data.copy()
    data['y']=raw_data['y'].map({'yes': 1,'no': 0})
    data=data.drop(['Unamed: 0'],axis=1)
    
    print(data)




if __name__ == '__main__':
    main()
