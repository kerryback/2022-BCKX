from numpy import *
import pandas as pd
from scipy.stats import norm
from scipy.stats import uniform
from scipy.integrate import quad
from scipy.integrate import dblquad
from scipy.optimize import root_scalar as root
from scipy.stats import multivariate_normal as binorm
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

''' read kappas (rows are times and columns are rhos) 
    new notation is z1 = kappa* '''

Kappa_star = pd.read_csv('Kappa_star.csv',index_col=0)
Kappa_star.columns = Kappa_star.columns.astype(float)
Kappa_star.columns = [round(x,3) for x in Kappa_star.columns]
Kappa_star.index = Kappa_star.index.astype(float)
Kappa_star.index = [round(x,3) for x in Kappa_star.index]

''' read parameters '''

params = pd.read_csv('params.csv',index_col=0)
mu = params.loc['mu'].item()
mustar = params.loc['mustar'].item()
sig = params.loc['sig'].item()

print('mu = ' + repr(mu) + ', mustar = ' + repr(mustar) + ', sig = ' + repr(sig))

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

rhos = array(Bdy.columns.to_list())
times = array(Bdy.index.to_list())

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
    
''' function z2 in new notation '''

def z1(t,u,y) :
    def f(z) :
        fn = t*norm.pdf(y) + (u-t)*norm.pdf(z) - z*(1-t*norm.cdf(y)-(u-t)*norm.cdf(z))
        first = t*norm.cdf(y) + (u-t)*norm.cdf(z) - 1
        second = (u-t)*norm.pdf(z)
        return  fn, first, second
    a = t*norm.pdf(y) / (1-t*norm.cdf(y))
    return root(f,x0=(a+z0(u))/2,fprime=True,fprime2=True,method='newton').root

''' linearly interpolate/extrapolate boundaries '''

def b(t,rho) :
    bdy = array(Bdy[rho].to_list())
    derivlow = (bdy[1]-bdy[0]) / (times[1]-times[0])
    derivhigh = (bdy[-1]-bdy[-2]) / (times[-1] - times[-2])
    if t < times[0] :
        return bdy[0]  + derivlow*(t-times[0])
    elif t > times[-1] :
        return bdy[-1] + derivhigh*(t-times[-1])
    else :
        cond = times<=t
        T1 = times[cond][-1] 
        T2 = times[~cond][0] 
        w = (T2-t) / (T2-T1)
        return w*Bdy.loc[T1,rho] + (1-w)*Bdy.loc[T2,rho]   
    
''' define pre-disclosure price '''

def pre_price(t,bdy,rho) :      # price at t prior to disclosure by firm 2
        kappa = (mustar-bdy)/sig
        delta = (1-rho)*kappa / sqrt(1-rho**2)
        num = t-t**2*norm.cdf(delta)
        cov = array([[1,rho],[rho,1]]) 
        Gamma = binorm.cdf(array([-kappa,-kappa]),cov=cov) 
        den = 1 - t**2 * (1 - Gamma) - 2*t*(1-t)*norm.cdf(kappa)
        return mustar - sig*(1+rho)*norm.pdf(kappa)*num/den
    
''' define post-disclosure price '''

# price at u when first firm discloses d at date t when boundary was b

def p(t,d,b,u,rho) :
    mustar1 = rho*d +(1-rho)*mustar
    sig1 = sig*sqrt(1-rho**2)
    y = (mustar1-b) / sig1
    z = z0(u)
    if y <= z:
        return mustar1 - sig1*z
    else :
        return mustar1 - sig1*z1(t,u,y)
    
''' read simulations '''

Sims = pd.read_csv('Sims.csv')
Sims.set_index(['rho','sim'],inplace=True)

newtimes = array(range(1,10)) / 10
newtimes = [0.01] + list(newtimes) + [0.99]
AvgPrice = pd.DataFrame(dtype=float,index=Sims.index,columns=newtimes) 
lastrho = 0
for rho, i in AvgPrice.index :
    if rho != lastrho : 
        print(rho)
        AvgPrice.to_csv('AvgPrice.csv')
    lastrho = rho
    x1 = Sims.loc[(rho,i),'x1']
    x2 = Sims.loc[(rho,i),'x2']
    avgx = (x1+x2) / 2
    tau1 = Sims.loc[(rho,i),'tau1']
    tau2 = Sims.loc[(rho,i),'tau2']
    for t in AvgPrice.columns :
        b1 = Bdy.loc[t,rho]
        p1 = Price.loc[t,rho]
        if t <= tau1 :
            AvgPrice.loc[(rho,i),t] = p1
        elif t >= tau2 :
            AvgPrice.loc[(rho,i),t] = avgx
        else :
            p2 = p(tau1,x1,b1,t,rho)
            AvgPrice.loc[(rho,i),t] = (x1+p2)/2
            
AvgPrice.to_csv('AvgPrice.csv')