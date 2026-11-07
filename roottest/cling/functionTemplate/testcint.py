from __future__ import print_function

import os
from pathlib import Path
from sys import stdout

import ROOT


def printme(o):
    print("t now %g %d %d" % (o.get["double"](), o.get["int"](), o.get["float"]()))
    stdout.flush()

# Use the library that the t-build test compiled into the working directory,
# instead of compiling t.h next to its source.
ROOT.gSystem.SetBuildDir(os.getcwd(), True)
ROOT.gROOT.ProcessLine(".L " + Path(__file__).resolve().with_name("t.h").as_posix() + "+")
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
