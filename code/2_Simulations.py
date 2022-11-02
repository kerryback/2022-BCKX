from numpy import *
import pandas as pd
from scipy.stats import norm
from scipy.stats import uniform
from scipy.optimize import root_scalar as root
from scipy.stats import multivariate_normal as binorm

''' read parameters '''

params = pd.read_csv('params.csv',index_col=0)
mu = params.loc['mu'].item()
mustar = params.loc['mustar'].item()
sig = params.loc['sig'].item()

''' read kappas (rows are times and columns are rhos)'''

Kappa_star = pd.read_csv('Kappa_star.csv',index_col=0)
Kappa_star.columns = [round(x,3) for x in Kappa_star.columns.astype(float)]
Kappa_star.index = [round(x,3) for x in Kappa_star.index.astype(float)]

''' define boundaries '''

Bdy = mustar - sig * Kappa_star

''' lists of correlations and times '''

rhos = array(Bdy.columns.to_list())
times = array(Bdy.index.to_list())

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
    
''' invert the linear interpolation/extrapolation '''

def t(x,rho) :
    bdy = array(Bdy[rho].to_list())
    derivlow = (bdy[1]-bdy[0]) / (times[1]-times[0])
    derivhigh = (bdy[-1]-bdy[-2]) / (times[-1] - times[-2])
    if x > bdy[0] :
        return max(0,times[0] + (x-bdy[0])/derivlow)
    elif x < bdy[-1] :
        return min(1,times[-1] + (x-bdy[-1])/derivhigh)
    else :
        cond = bdy>x
        B1 = bdy[cond][-1]
        B2 = bdy[~cond][0]
        w = (B2-x) / (B2-B1)
        Indx = pd.Index(Bdy[rho])
        i1 = Indx.get_loc(B1)
        i2 = Indx.get_loc(B2)
        T1 = times[i1]
        T2 = times[i2]
        return w*T1 + (1-w)*T2
    
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

''' price at u when first firm discloses d at date t when boundary was b '''

def p(t,d,b,u,rho) :
    mustar1 = rho*d +(1-rho)*mustar
    sig1 = sig*sqrt(1-rho**2)
    y = (mustar1-b) / sig1
    z = z0(u)
    if y <= z:
        return mustar1 - sig1*z
    else :
        return mustar1 - sig1*z1(t,u,y)
    
''' announcement time for second firm to disclose '''  

# date of 2nd firm's disclosure, given first firm discloses d at date t when boundary was b
# and 2nd firm's value is x

def u(t,d,b,x,rho) :
    mustar1 = rho*d +(1-rho)*mustar
    sig1 = sig*sqrt(1-rho**2)
    z = (mustar1-x)/sig1
    if x < b :
        return max(t, z / (norm.pdf(z) + z*norm.cdf(z)))
    else :
        y = (mustar1-b)/sig1
        num = z*(1-t*norm.cdf(y)+t*norm.cdf(z)) - t*norm.pdf(y) + t*norm.pdf(z)
        return num / (norm.pdf(z) + z*norm.cdf(z))   
    
''' announcement times for both firms '''

def announcements(ta,tb,xa,xb,rho) :         # a is the first firm to get signal
    if xa >= b(ta,rho) :                     # firm a discloses when it receives signal
        timea = ta
        if xb > p(ta,xa,b(ta,rho),tb,rho) :  # firm b discloses when it receives signal
             timeb = tb
        else :                               # firm b discloses later
            timeb = u(ta,xa,b(ta,rho),xb,rho)
    elif xa >= b(tb,rho) :                   # firm a discloses first, after it receives signal
        timea = t(xa,rho)                    #     but before firm b receives its signal
        if xb > p(timea,xa,xa,tb,rho) :      # firm b discloses when it receives signal
            timeb = tb
        else :                               # firm b discloses later
            timeb = u(timea,xa,xa,xb,rho) 
    elif xa > xb :                           # firm a discloses after firm b gets its signal
        timea = t(xa,rho)
        timeb = u(timea,xa,xa,xb,rho) 
    elif xb >= b(tb,rho) :                   # firm b discloses first, when it receives signal
        timeb = tb  
        timea = u(tb,xb,b(tb,rho),xa,rho)  
    else :
        timeb = t(xb,rho)                    # firm b discloses first, after it receives signal 
        timea = u(timeb,xb,xb,xa,rho) 
    return timea, timeb                      # announcement time of first firm to get signal, then second firm to get signal

''' generate random times - firm1 is the first firm to get its signal
    maybe more work here than needed.  this ensures even distribution of random times across 10x10 blocks of
    [0,1] x [0,1].  numsims should be a multiple of 100 '''

def randomtimes(numsims) :
    size = int(numsims/100)
    df = pd.DataFrame(dtype=float,columns=[1,2])
    for i in range(10) :
        for j in range(10) :
            rvs1 = i/10 + (1/10)*uniform.rvs(size=size)
            rvs2 = j/10 + (1/10)*uniform.rvs(size=size)
            rvs = pd.DataFrame({1:rvs1,2:rvs2},index=i*10*size+ j*size + arange(size))
            df = pd.concat((df,rvs))
    df['theta1'] = [min(x,y) for x,y in zip(df[1],df[2])]
    df['theta2'] = [max(x,y) for x,y in zip(df[1],df[2])]
    return df.drop(columns=[1,2])

''' generate random values for a given correlation and combine with random times '''

def randomvalues(rtimes,rho) :
    numsims = rtimes.shape[0]
    corr = array([[1,rho],[rho,1]])
    rx = binorm.rvs(mean=[mu,mu],cov=sig**2*corr,size=numsims)
    df = pd.DataFrame(rx,index=range(numsims),columns=['x1','x2'])
    return rtimes.join(df)

''' add announcement dates - firm1 is still first firm to get signal 
    theta is signal date, tau is disclosure date '''

def announce(sims,rho) :
    out = sims
    out['tau1'] = nan    # disclosure time of first firm to get signal
    out['tau2'] = nan    # disclosure time of second firm to get signal
    out['ind1'] = nan
    out['ind2'] = nan
    for i in sims.index :
        ta = out.loc[i,'theta1']
        tb = out.loc[i,'theta2']
        xa = out.loc[i,'x1']
        xb = out.loc[i,'x2']
        out.loc[i,'tau1'], out.loc[i,'tau2'] = announcements(ta,tb,xa,xb,rho)
    return out    

''' rename firms so that firm 1 is the first firm to disclose '''

def reorder(sims) :
    out = sims.copy()
    cond = (out['tau1']<out['tau2']) | ( (out['tau1']==out['tau2']) & (out['x1']>=out['x2']) )    
    out['x1'] = where(cond,sims['x1'],sims['x2'])
    out['x2'] = where(cond,sims['x2'],sims['x1'])
    out['theta1'] = where(cond,sims['theta1'],sims['theta2'])
    out['theta2'] = where(cond,sims['theta2'],sims['theta1'])
    out['tau1'] = where(cond,sims['tau1'],sims['tau2'])
    out['tau2'] = where(cond,sims['tau2'],sims['tau1'])
    return out
    
''' create simulations '''

def initialize(numsims,corrs) :
    rtimes = randomtimes(numsims)
    count = 0
    for rho in corrs :
        print(rho)
        df = randomvalues(rtimes,rho)
        df = announce(df,rho)
        df = reorder(df)
        df['rho'] = rho
        df['sim'] = range(numsims)
        df = df.reset_index(drop=True).set_index(['rho','sim'])
        if count == 0 :
            final = df.copy()
        else :
            final = pd.concat((final,df))
        count += 1
    return final    

''' add boundaries and prices 
    B1 is Stage 1 boundary, P1 is Stage 1 price, P2 is Stage 2 price & bdy '''

def finish(sims) :
    final = sims
    final['ind1'] = (final['tau1']!=final['theta1'])  # True means disclosing after information arrival
    final['ind2'] = (final['tau2']!=final['theta2'])
    for col in ['B1_tau1','P1_tau1','P2_tau1','P2_tau2'] :
        final[col] = nan
    lastrho = 0
    for rho, i in final.index :
        if rho != lastrho : 
            print(rho)
            final.to_csv('Sims.csv')
        lastrho = rho
        tau1 = final.loc[(rho,i),'tau1']
        x1 = final.loc[(rho,i),'x1']
        x2 = final.loc[(rho,i),'x2']
        if final.loc[(rho,i),'ind1'] == 0 :
            final.loc[(rho,i),'B1_tau1'] = x1
            final.loc[(rho,i),'P1_tau1'] = pre_price(tau1,x1,rho)
            final.loc[(rho,i),'P2_tau1'] = p(tau1,x1,x1,tau1,rho)
            if final.loc[(rho,i),'ind2'] :
                final.loc[(rho,i),'P2_tau2'] = x2
            else :
                tau2 = final.loc[(rho,i),'tau2']
                final.loc[(rho,i),'P2_tau2'] = p(tau1,x1,x1,tau2,rho)
        else :
            b1 = b(tau1,rho)
            final.loc[(rho,i),'B1_tau1'] = b1
            final.loc[(rho,i),'P1_tau1'] = pre_price(tau1,b1,rho)
            final.loc[(rho,i),'P2_tau1'] = p(tau1,x1,b1,tau1,rho)
            if final.loc[(rho,i),'ind2'] :
                final.loc[(rho,i),'P2_tau2'] = x2
            else :
                tau2 = final.loc[(rho,i),'tau2']
                final.loc[(rho,i),'P2_tau2'] = p(tau1,x1,b1,tau2,rho)
    return final.drop(columns=['ind1','ind2'])

''' run '''

numsims = 100000
sims = initialize(numsims,rhos)
sims = finish(sims)
sims.to_csv('Sims.csv')

