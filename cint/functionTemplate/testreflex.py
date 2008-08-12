from ROOT import *;

def printme(o):
    print "t now %g %d %d" % (o.get(Double)(), o.get(int)(), o.get(float)());
    print "t now %g" % (o.getfloat());

gSystem.Load("libCintex");
Cintex.Enable();
gSystem.Load("libt_rflx.so");
print dir(t);
o = t();
printme(o);
o.set(12);
printme(o);
o.set(42.34);
printme(o);


