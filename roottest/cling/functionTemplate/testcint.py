from __future__ import print_function

from sys import stdout

import ROOT


def printme(o):
    print("t now %g %d %d" % (o.get["double"](), o.get["int"](), o.get["float"]()))
    stdout.flush()

ROOT.gROOT.ProcessLine(".L t.h+")
sortedMethods = [ item for item in ROOT.t.__dict__.keys() if item[0:2] != '__' ]
sortedMethods.sort()
print("# just a comment")
print(sortedMethods)
stdout.flush()
o = ROOT.t()
printme(o)
o.set(12)
printme(o)
o.set(42.34)
printme(o)
