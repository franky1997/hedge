#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 23:09:36 2017

@author: jia_qu
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels import regression
import statsmodels.api as sm
from scipy import stats
# helper loader function
loader = lambda asset: np.loadtxt('%s.txt' % asset, delimiter=',')

# raw data
spy = loader('SPY')
aapl = loader('AAPL')
goog = loader('GOOG')

# normalised data
spy_ = spy / spy[0] - 1
aapl_ = aapl / aapl[0] - 1
goog_ = goog / goog[0] - 1
"""
# visualization
plt.figure()
plt.plot(spy_, label='S&P500')
plt.plot(aapl_, label='Apple Inc.')
plt.plot(goog_, label='Google Inc.')
plt.title('Normalised Prices')
plt.ylabel('Normalised Change')
plt.xlabel('Time Index')
plt.legend();
"""
"""
# regression plots
plt.figure()
sns.regplot(spy_, aapl_, label='APPL')
sns.regplot(spy_, goog_, label='GOOG')
plt.title('Beta Exposure to S&P500')
plt.xlabel('SPY Percentage Change')
plt.ylabel('Stock Price Percentage Change')
plt.legend();
"""
# helper percentage return
pct_change = lambda series: np.diff(series) / (series[:-1] + 1e-6) # WARNING: prone to division by zero

aapl_pct = pct_change(aapl)
goog_pct = pct_change(goog)
spy_pct = pct_change(spy)
"""
plt.figure()
sns.distplot(aapl_pct, label='AAPL')
sns.distplot(goog_pct, label='GOOG')
sns.distplot(spy_pct, label='SPY')
plt.legend();
"""
def datapoints(x):
    """Returns a list Converts data points into the form (x,1)"""
    X=[] #X is an array of 10 points of form (x,1)
    for i in range(len(x)):
        X.append(np.array([x[i],1]))
    X=np.asarray(X)
    return X

spy_1=datapoints(spy_)
#aapl_=datapoints(aapl_)
#goog_=datapoints(goog_)

def graddescend(x,y,n,eta):
    """Returns an array (beta,alpha). Require array(x),array(y),n=number of iterations,eta=size of change"""
    w = np.random.random(x.shape[1])
    for i in range(n):
        y_hat=np.dot(x,w)
        i=np.random.randint(len(y))
        error=(y_hat[i]-y[i])
        w=w-eta*error*x[i]
    return w


def linreg(x,y):
    x = sm.add_constant(x)
    model = regression.linear_model.OLS(y,x).fit()
    
    return model.params[0], model.params[1]

def regression(x,y):
    model=stats.linregress(x,y)
    return model[0],model[1]
regression(spy_,aapl_)



def portfolio(stock1,stock2,benchmark):
    """calculate the portfolio, input the price of two stocks and the benchmark as arrays"""
    beta1=regression(benchmark,stock1)
    beta2=regression(benchmark,stock2)
    gamma=(beta1[0]/(beta1[0]-beta2[0]))
    lamb=1-gamma
    
    P=[]
    for i in range(spy_1.shape[0]):
        portfolio=lamb*np.dot(spy_1[i],beta1)+gamma*np.dot(spy_1[i],beta2)
        P.append(portfolio)
    return P

P=portfolio(aapl_,goog_,spy_)
stats=regression(spy_,P)
beta_n, alpha_n= stats
print 'Portfolio Output:'
print 'alpha: ' + str(alpha_n)
print 'beta: ' + str(beta_n)

#print statistics.summary()

plt.figure()
plt.plot(spy_1[:,0], label='S&P500')
plt.plot(aapl_, label='Apple Inc.')
plt.plot(goog_, label='Google Inc.')
plt.plot(P,label='beta hedge')
plt.title('Normalised Prices')
plt.ylabel('Normalised Change')
plt.xlabel('Time Index')
plt.legend();
