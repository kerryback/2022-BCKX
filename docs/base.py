from numpy import *
import pandas as pd
from scipy.stats import norm
from scipy.stats import uniform
from scipy.integrate import quad
from scipy.integrate import dblquad
from scipy.optimize import root_scalar as root
from scipy.stats import multivariate_normal as binorm
import plotly.graph_objects as go

''' read kappas (rows are times and columns are rhos) 
    new notation is z1 = kappa* '''

Kappa_star = pd.read_csv('../code/Kappa_star.csv',index_col=0)
Kappa_star.columns = Kappa_star.columns.astype(float)
Kappa_star.columns = [round(x,3) for x in Kappa_star.columns]
Kappa_star.index = Kappa_star.index.astype(float)
Kappa_star.index = [round(x,3) for x in Kappa_star.index]

''' read parameters '''

params = pd.read_csv('../code/params.csv',index_col=0)
mu = params.loc['mu'].item()
mustar = params.loc['mustar'].item()
sig = params.loc['sig'].item()

''' define boundaries and physical-probability kappas '''

Bdy = mustar - sig * Kappa_star
Kappa = (mu - Bdy) / sig

''' prices and physical expectations '''

def pre_mean(mean,bdy,t,rho) :  # mean = mu or mustar
    kappa = (mean-bdy)/sig
    delta = (1-rho)*kappa / sqrt(1-rho**2)
    cov = array([[1,rho],[rho,1]]) 
    num =  (1+rho)*norm.pdf(kappa)*(t- t**2*norm.cdf(delta))
    Gamma = binorm.cdf(array([-kappa,-kappa]),cov=cov)
    den = 1 - t**2*(1-Gamma) - 2*t*(1-t)*norm.cdf(kappa)
    return mean - sig*num / den

Mean = pd.DataFrame(dtype=float,index=Bdy.index,columns=Bdy.columns)
for t in Bdy.index :
    for rho in Bdy.columns :
        bdy = Bdy.loc[t,rho]
        Mean.loc[t,rho] = pre_mean(mu,bdy,t,rho)

Price = pd.DataFrame(dtype=float,index=Bdy.index,columns=Bdy.columns)
for t in Bdy.index :
    for rho in Bdy.columns :
        bdy = Bdy.loc[t,rho]
        Price.loc[t,rho] = pre_mean(mustar,bdy,t,rho)

''' lists of correlations and times '''

rhos = Bdy.columns.to_list()
times = Bdy.index.to_list()
Times = pd.Series(times,index=times)  # used this later, probably shouldn't have

''' function z in new notation, can input list of times '''

def z0(u) :
    if isinstance(u,list) :
        return array([z0(x) for x in u])
    else :
        def f(z) :  
            fn = u*norm.pdf(z) - z*(1-u*norm.cdf(z))
            first = u*norm.cdf(z) - 1
            second = u*norm.pdf(z) 
            return fn, first, second
        return root(f,x0=1,fprime=True,fprime2=True,method='newton').root

import numpy as np
mustar = -0.5

def price(t,mu) :
    return mu-sig*z0(t)

def pdf(p,mu) :
    return norm.pdf((p-mu)/sig)

def cdf(p,mu) :
    return norm.cdf((p-mu)/sig)

def prob1(t,mu) :
    p = price(t,mustar)
    return t*pdf(p,mu) / (t*pdf(p,mu) + 1 - cdf(p,mu))

def prob2(t,mu) :
    p = price(t,mustar)
    return (1-cdf(p,mu)) / (t*pdf(p,mu) + 1 - cdf(p,mu))

def condmean(t,mu) :
    p = price(t,mustar)
    return mu + sig*pdf(p,mu)/(1-cdf(p,mu))


def mean(t,mu) :
    p = price(t,mustar)
    return prob1(t,mu)*p + prob2(t,mu)*condmean(t,mu)

mu = 105
mustar = 100
sig = 15

''' function z2 in new notation '''

def z1(t,u,y) :
    def f(z) :
        fn = t*norm.pdf(y) + (u-t)*norm.pdf(z) - z*(1-t*norm.cdf(y)-(u-t)*norm.cdf(z))
        first = t*norm.cdf(y) + (u-t)*norm.cdf(z) - 1
        second = (u-t)*norm.pdf(z)
        return  fn, first, second
    a = t*norm.pdf(y) / (1-t*norm.cdf(y))
    return root(f,x0=(a+z0(u))/2,fprime=True,fprime2=True,method='newton').root

''' firm 1 price after firm 2 disclosure (old notation: firm 2 discloses first) 
    risk neutral expectation of x1 after disclosure of x2 
    computed as fixed point '''

def firm1_price(t,rho,x2,u) : 
    bdyt = Bdy.loc[t,rho]
    mustar1 = rho*x2 + (1-rho)*mustar
    sig1 = sig * sqrt(1-rho**2)
    y = (mustar1-bdyt) / sig1
    z = z0(u)
    if y <= z:
        return mustar1 - sig1*z
    else :
        return mustar1 - sig1*z1(t,u,y)

def firm1_expected(t,rho,x2,u) :
    bdyt = Bdy.loc[t,rho]
    mu1 = rho*x2 + (1-rho)*mu
    sig1 = sig * sqrt(1-rho**2)
    bdyu = firm1_price(t,rho,x2,u)   # price and bdy are same in Stage 2
    if bdyu <= bdyt :
        a = (mu1-bdyu) / sig1
        return mu1 - sig1 * u*norm.pdf(a) / (1-u*norm.cdf(a))
    else :
        a_t = (mu1-bdyt) / sig1
        a_u = (mu1-bdyu) / sig1
        return mu1 - sig1 * ( (t*norm.pdf(a_t)+(u-t)*norm.pdf(a_u)) / (1-t*norm.cdf(a_t)-(u-t)*norm.cdf(a_u)) ) 