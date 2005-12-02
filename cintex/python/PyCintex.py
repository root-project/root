## system modules
import os, sys, exceptions, string

import ROOT
from libPyROOT import makeRootClass

ROOT.SetMemoryPolicy( ROOT.kMemoryStrict )
## Load Cintex module and enable conversions Reflex->CINT

def loadDict( dict ) :
  if sys.platform == 'win32' :
    ROOT.gSystem.Load(dict)
  else :
    ROOT.gSystem.Load('lib' + dict)

loadDict('lcg_Cintex')
ROOT.Cintex.setDebug(0)
ROOT.Cintex.enable()

def makeNamespace( name ) :
  if name == '' : return ROOT
  else :          return getattr(ROOT,name)

def makeClass( name ) :
  return makeRootClass(name)
  
def addressOf( obj ) :
  return ROOT.AddressOf(obj)[0]
       
def getAllClasses( ) :
  TClassTable = makeRootClass('TClassTable')
  TClassTable.Init()
  classes = []
  while True :
    c = TClassTable.Next()
    if c : classes.append(c)
    else : break
  return classes

## Global namespace
gbl  = makeNamespace('')
NULL = ROOT.NULL

if __name__ == '__main__' :
  loadDict('test_CintexDict')

    