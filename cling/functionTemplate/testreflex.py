from ROOT import gSystem, Cintex
from sys import stdout

def printme(o):
    print "t now %g %d %d" % (o.get(Double)(), o.get(int)(), o.get(float)())
    print "t now %g" % (o.getfloat())
    stdout.flush()

gSystem.Load("libCintex")
Cintex.Enable()
gSystem.Load("t_rflx_wrap_cxx")
sortedMethods = [ item for item in t.__dict__.keys() if item[0:2] != '__' ]
sortedMethods.sort()
print "# just a comment"
print sortedMethods
stdout.flush()
o = t()
printme(o)
o.set(12)
printme(o)
o.set(42.34)
printme(o)


