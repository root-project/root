from ROOT import *
from sys import stdout

def printme(o):
    print "t now %g %d %d" % (o.get("double")(), o.get("int")(), o.get("float")())
    stdout.flush() 

#gSystem.Load("libG__t.so")
gROOT.ProcessLine(".L t.h+")
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
