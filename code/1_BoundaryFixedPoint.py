from numpy import *
import pandas as pd
from scipy.stats import norm
from scipy.stats import multivariate_normal as binorm
from scipy.integrate import dblquad
from scipy.optimize import root_scalar as root

def z0(u) :
    def f(z) :  
        fn = u*norm.pdf(z) - z*(1-u*norm.cdf(z))
        first = u*norm.cdf(z) - 1
        second = u*norm.pdf(z) 
        return fn, first, second
    return root(f,x0=1,fprime=True,fprime2=True,method='newton').root

def z1(t,u,y) :
    def f(z) :
        fn = t*norm.pdf(y) + (u-t)*norm.pdf(z) - z*(1-t*norm.cdf(y)-(u-t)*norm.cdf(z))
        first = t*norm.cdf(y) + (u-t)*norm.cdf(z) - 1
        second = (u-t)*norm.pdf(z)
        return  fn, first, second
    a = t*norm.pdf(y) / (1-t*norm.cdf(y))
    return root(f,x0=(a+z0(u))/2,fprime=True,fprime2=True,method='newton').root

def kappa(rho,t) :
    def foc(k) :
        d = (1-rho)*k/sqrt(1-rho**2)
        cov = array([[1,rho],[rho,1]])
        num = (1+rho)*norm.pdf(k)*(t-t**2*norm.cdf(d))
        Gamma = binorm.cdf(array([-k,-k]),cov=cov)
        den = 1 - t**2*(1-Gamma) - 2*t*(1-t)*norm.cdf(k)
        lhs = rho*(num/den - k)
        def integrand(u,y) :
            return (y-z1(t,u,y))*norm.pdf((y-(1+rho)*d)/rho)
        integral = dblquad(integrand,z0(t),inf,lambda x: t,lambda x: x/(norm.pdf(x)+x*norm.cdf(x)))[0]
        rhs = sqrt(1-rho**2)*integral / (1 - t*norm.cdf(d))
        return lhs - rhs   
    return root(foc,x0=-1,x1=1,method='secant').root

times = array(range(1,100))/100
rhos = array(range(1,20))/20
rhos = [0.01] + list(rhos) + [0.99]
# rhos = [0.99]
# df = pd.DataFrame(dtype=float,index=times,columns=rhos)
for rho in rhos :
    ser = pd.Series(dtype=float,index=times)
    for t in times :
        ser[t] = kappa(rho,t)
        print('rho = ' + repr(rho) + ', t = ' + repr(t))
    ser.to_csv('kappa_' + repr(rho) + '.csv')

        
