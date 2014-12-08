#
# Module for reading TSAL files.
#
# This is really supposed to be an auxiliary 
# class geared at plotting, although it can 
# do other basic manipulations of tsals
#

from scipy import *

class parameter(object):
    def __init__(self, name, value=None, error=None):
        self.name=name
        self.val=value
        self.err=error

    def setParameterLine(self,line):
        self.line=line
  
class TSAL(object):
    def __init__ (self, fname,lastColIsX=False):
        ff=open(fname)
        self.readTSAL (ff,lastColIsX)
        ff.close()

    def readTSAL (self, ff,lastColIsX=False):
        d = ff.readline().split()
        version = d[0]
        N = d[1]
        # if (version!="v0"):
        #     print "Ehm, version not v0, we bravely go ahead."
        N=int(N)
        pars={}
        if lastColIsX:
            xvals=[]
            yvals=[]
            yerrs=[]
        for i in range(N):
            line=ff.readline()
            mean, err, name = line.split()[:3]
            mean=float(mean)
            err=float(err)
            par=parameter(name,mean,err)
            par.setParameterLine(line) ## maybe somebody can do something useful
            pars[name]=par

            if lastColIsX:
                xvals.append(float(line.split()[-1]))
                yvals.append(mean)
                yerrs.append(err)

        self.pars=pars
        self.fd=map(float, [ff.readline() for i in range(N)])
        self.sd=self.readMatrix(ff)
        if lastColIsX:
            self.xvals=array(xvals)
            self.yvals=array(yvals)
            self.yerrs=array(yerrs)
        

    def readMatrix(self,ff):
        line=ff.readline()
        ty, N=line.split()
        N=int(N)
        if ty=="DenseSymMatrix":
            mat=array(map(lambda x:map(float,x.split()),map(lambda x:ff.readline(), range(N))))
        elif ty=="DiagonalMatrix":
            mat=zeros((N,N))
            for i in range(N):
                mat[i,i]=float(ff.readline())
        else:
            print "Dont recognize matrix"
            stop()
        self.mat=mat

