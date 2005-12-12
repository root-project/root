""" PyCintex emulation module.
    This module emulates the functionality provided by PyLCGDict and PyLCGDict2
    implementations of Python bindings using the Reflection dictionaries.
    The current implementation is based on Reflex dictionaries with PyROOT.
    PyROOT provides the basic pyhton bindings to C++ and Cintex to populate 
    CINT dictionaries from Reflex ones.
"""
#---- system modules--------------------------------------------------
import os, sys, exceptions, string

#---- Import PyROOT module -------------------------------------------
import ROOT
from libPyROOT import makeRootClass
ROOT.SetMemoryPolicy( ROOT.kMemoryStrict )

#--- LoadDictionary function and aliases -----------------------------
def loadDictionary(name) :
  if sys.platform == 'win32' :
    sc = ROOT.gSystem.Load(name)
  else :
    sc = ROOT.gSystem.Load(name)
  if sc == -1 : raise "Error Loading dictionary" 
loadDict = loadDictionary

#--- Load Cintex module and enable conversions Reflex->CINT-----------
ROOT.gSystem.Load('libReflex')
ROOT.gSystem.Load('libCintex')
ROOT.Cintex.SetDebug(0)
ROOT.Cintex.Enable()

#--- Other functions needed -------------------------------------------
def Namespace( name ) :
  if name == '' : return ROOT
  else :          return getattr(ROOT,name)
makeNamespace = Namespace

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

#--- Global namespace and global objects -------------------------------
gbl  = makeNamespace('')
NULL = ROOT.NULL
class double(float): pass
class short(int): pass
class long_int(int): pass
class unsigned_short(int): pass
class unsigned_int(int): pass
class unsigned_long(long): pass

#--- For test purposes --------------------------------------------------
if __name__ == '__main__' :
  loadDict('test_CintexDict')

    
