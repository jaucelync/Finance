# -*- coding: utf-8 -*-
"""
Created on Sat Sep  7 11:19:19 201
@author: Jaucelyn
"""

import numpy as np
import pandas
import math 
from matplotlib import pyplot as plt
from scipy import stats
import statsmodels.api as sm 
from statsmodels.graphics.gofplots import qqplot


data = pandas.read_excel('JaucelynFinance.xlsx')
value = data.values

#years list
#t = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44])
VFINX_price = np.array(value[2:45,1]).tolist()
VFINX_price.reverse()
VFINX_price = np.array(VFINX_price)
VFINX_dy = np.array(value[2:44, 2]).tolist()
VFINX_dy.reverse()
VFINX_dy = np.array(VFINX_dy)
VWESX_price = np.array(value[2:45, 3]).tolist()
VWESX_price.reverse()
VWESX_price = np.array(VWESX_price)
VWESX_dy = np.array(value[2:44, 4]).tolist()
VWESX_dy.reverse()
VWESX_dy = np.array(VWESX_dy)
VMMXX_price = np.array(value[2:45,5]).tolist()
VMMXX_price.reverse()
VMMXX_price = np.array(VMMXX_price)
VMMXX_dy = np.array(value[2:44, 6]).tolist()
VMMXX_dy.reverse()
VMMXX_dy = np.array(VMMXX_dy)
#total annual return of VFINX
tot = [math.log(VFINX_price[t+1] / VFINX_price[t]) + math.log(1 + VFINX_dy[t]) for t in range(42)]

#equity premium
equ = [tot[t] - math.log(1 + VWESX_dy[t]) for t in range(42)]

#pyplot.hist(equ, bins=20)
#pyplot.show(equ)


#apply summing-in-3 operation to ln(1 + MMF) as well, get 3-year bond returns

stocks = tot
bonds = [math.log(VWESX_price[t+1]/VWESX_price[t]) + math.log(1 + VWESX_dy[t]) for t in range(42)]
VMMXX = [math.log(VMMXX_price[t+1]/VMMXX_price[t]) + math.log(1 + VMMXX_dy[t]) for t in range(42)] # i am not sure if this was how I was to compute the returns, considering the price is always 1

stocks_3 = np.array([sum(stocks[i:i+3]) for i in range(0, len(stocks), 3)])
bonds_3 = np.array([sum(bonds[i:i+3]) for i in range(0, len(bonds), 3)])
VMMXX_3 = np.array([sum(VMMXX[i:i+3]) for i in range(0, len(VMMXX), 3)])

stock_premium = stocks_3 - VMMXX_3
bonds_premium = bonds_3 - VMMXX_3

stocks_mean = np.mean(stock_premium)
stocks_sd = np.std(stock_premium)

bonds_mean = np.mean(bonds_premium)
bonds_sd = np.std(bonds_premium)

corr = np.correlate(stock_premium, bonds_premium)

#create matrix to calculate the covariance matrix
mat = np.stack((stock_premium,bonds_premium),axis=0)

#covariance = np.cov(mat)
covariance = np.cov(stock_premium, bonds_premium)
#combine the means for the multivariate_normal function
combined_mean = np.array([stocks_mean, bonds_mean])
NSIMS = 10000
#now we can use the multivariate_normal function
normSim = np.random.multivariate_normal(combined_mean, covariance, NSIMS)



def portfolio(ret, f):
    stock_split = f * math.exp(ret[0])
    bond_split = (1-f) * math.exp(ret[1])
    total = stock_split + bond_split
    return total;

def portfolio_add(ret, f, extra):
    stock_split = f * math.exp(ret[0])
    bond_split = (1-f) * math.exp(ret[1])
    stock = (f * extra) + stock_split
    bond = ((1-f) * extra) + bond_split 
    total = stock + bond
    return total;




returns_mat = np.empty((21, NSIMS))
for k in range(21):
    for i in range(NSIMS):
        returns_mat[k,i] = portfolio(normSim[i],k*.05)

  
sequence = [2+n for n in range(0,NSIMS,3)]

newreturns_mat = np.empty((21, NSIMS))
for k in range(21):
    for i in range(NSIMS): 
        if i in sequence:
            newreturns_mat[k,i] = portfolio_add(normSim[i],k*.05,1)
        else:
            newreturns_mat[k,i] = portfolio(normSim[i],k*.05)


def withdraw(ele,frac):
    wealth_withdrawn = ele * frac
    leftover = ele - wealth_withdrawn
    combined = np.array([leftover,wealth_withdrawn])
    return combined;

wealth_withdrew = np.empty((21,len(sequence) * 3))
redreturns_mat = np.empty((21, NSIMS)) 
for k in range(21):
    for i in range(NSIMS): 
        if i in sequence:
            call_withdraw = withdraw(returns_mat[k,i],.12)
            redreturns_mat[k,i] = call_withdraw[0]
            wealth_withdrew[k,i] = call_withdraw[1]
        else:
            redreturns_mat[k,i] = returns_mat[k,i]
            
wealth_withdrew = wealth_withdrew[wealth_withdrew != 0]
wealth_withdrew = wealth_withdrew.reshape(21,len(sequence)-1)
wealth_withdrew_summed = wealth_withdrew.sum(axis=1) #what will i do with this now

def Multiply_10(returns):
    result = 1
    for i in range(10):
        result *=  returns[i]
    return result 
 
sim_mat = np.empty((21, 1000))
for k in range(21):
    for i in range(1000):
        sim_mat[k,i] = np.array(Multiply_10(returns_mat[k,i*10:i*10+10]))
        
redsim_mat = np.empty((21, 1000))
for k in range(21):
    for i in range(1000):
        redsim_mat[k,i] = np.array(Multiply_10(redreturns_mat[k,i*10:i*10+10]))



sim_mean = np.array([np.mean(sim_mat[i]) for i in range(21)])
sim_std = np.array([np.std(sim_mat[i]) for i  in range(21)])
VaR_95_sim = np.array([stats.norm.ppf(0.05, sim_mean, sim_std)])

redsim_mean = np.array([np.mean(redsim_mat[i]) for i in range(21)])
redsim_std = np.array([np.std(redsim_mat[i]) for i  in range(21)])
redVaR_95_sim = np.array([stats.norm.ppf(0.05, redsim_mean, redsim_std)])

plt.bar(range(21),sim_mean)
plt.bar(range(21),redsim_mean) 

#for i in range(21):
    #plt.hist(sim_mat[i],alpha = 0.5, bins = 100)
    
#returns_60_40 = [portfolio(normSim[i], 0.6) for i in range(NSIMS)] 
#returns_100_0 = [portfolio(normSim[i], 1) for i in range(NSIMS)] 
#returns_95_5 = [portfolio(normSim[i], .95) for i in range(NSIMS)]
#returns_90_10 = [portfolio(normSim[i], .9) for i in range(NSIMS)]
#returns_10_90 = [portfolio(normSim[i], .1) for i in range(NSIMS)]
#returns_5_95 = [portfolio(normSim[i], .5) for i in range(NSIMS)]
#returns_0_100 = [portfolio(normSim[i], 0) for i in range(NSIMS)]


#sim_60_40 = np.array([Multiply_10(returns_60_40[i:i+10]) for i in range(0, len(returns_60_40), 10)])
#sim_100_0 = np.array([Multiply_10(returns_100_0[i:i+10]) for i in range(0, len(returns_100_0), 10)])
#sim_95_5 = np.array([Multiply_10(returns_95_5[i:i+10]) for i in range(0, len(returns_95_5), 10)])
#sim_90_10 = np.array([Multiply_10(returns_90_10[i:i+10]) for i in range(0, len(returns_90_10), 10)])
#sim_10_90 = np.array([Multiply_10(returns_10_90[i:i+10]) for i in range(0, len(returns_10_90), 10)])
#sim_5_95 = np.array([Multiply_10(returns_5_95[i:i+10]) for i in range(0, len(returns_5_95), 10)])
#sim_0_100 = np.array([Multiply_10(returns_0_100[i:i+10]) for i in range(0, len(returns_0_100), 10)])




##def test_portfolio(nsims):
   # x = 1
   # x_list = []
    #for count in range(nsims):
     #  x = 
      # x_list[i] = x   
        
   #return x

#returns_1000 = [test_portfolio(1000)]
#returns = np.array(returns)

    
#pyplot.hist(returns, bins=150) 

#qqplot(stocks_3, line = 's') 
#qqplot(bonds_3, line = 's')




