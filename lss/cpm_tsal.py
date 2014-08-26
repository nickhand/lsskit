#
# Module for reading TSAL files, ComovingPowerMeasurement version
  
from .tsal import *
import pylab
import sys

class CPM_TSAL(TSAL):
    def __init__ (self, fname):
        ff=open(fname)
        self.readTSAL (ff)
        self.readExtra(ff)
        ff.close()

    def readExtra (self,ff):
      #read redshift
      z=ff.readline()
      line=ff.readline()
      if line!="add_to_baseline\n" :
        print line
        print "no add_to_baseline - Not sure what to do"
        sys.exit()
      baseline={}
      ks=set()
      mus=set()
      N=int(ff.readline())
      for i in range(N):
        k,mu,val=map(float,ff.readline().split())
        ks.add(k)
        mus.add(mu)
        baseline[(k,mu)]=val
      self.baseline=baseline
      self.ks=ks
      self.mus=mus
      line=ff.readline()
      if line!="approx_noise\n" :
        print line
        print "no approx_noise - Not sure what to do"
        sys.exit()
      noise={}
      if N != int(ff.readline()) :
        print "number mismatch"
        sys.exit()
      for i in range(N):
        k,mu,val=map(float,ff.readline().split())
        if (k,mu) not in self.baseline :
          print i, k, mu
          print "k or mu mismatch"
          sys.exit()
        noise[(k,mu)]=val
      self.noise=noise

    def getMeasurement(self,k,mu):
      if (k,mu) not in self.baseline:
        print "What you request is not in baseline"
      base=self.baseline[(k,mu)]
      noise=self.noise[(k,mu)]
      name="Pkmu_X_"+str(k)+"_"+str(mu)
      val=base+self.pars[name].val
      err=self.pars[name].err
      return val,err,noise,base

    def plotFractionalError(self,k,mu,fmt='bo'):
      val,err,noise,base=self.getMeasurement(k,mu)
      pylab.semilogy(k,err/(val-noise),fmt)

    def plotMeasurement(self,k,mu,fmt='k-'):
      val,err,noise,base=self.getMeasurement(k,mu)
      pylab.loglog(k,val-noise,fmt)
      #pylab.semilogy(k,val-noise)
      #points come in here one at a time so can't really plot line
      pylab.errorbar(k,val-noise,yerr=err,fmt=fmt)
      #pylab.loglog(k,base-noise,'k-')
            
    def aggregateErrors(self,kmin,kmax,mumin,mumax) :
      s=0
      for i,mu in enumerate(self.mus):
        for j,k in enumerate(self.ks):
          if k>kmin and k<kmax and mu>mumin and mu<mumax : 
            val,err,noise,ignore=self.getMeasurement(k,mu)
            s+=((val-noise)/err)**2
      print kmin,kmax,mumin,mumax, 1/sqrt(s)
 
