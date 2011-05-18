""" PyCintex emulation module.
    This module emulates the functionality provided by PyLCGDict and PyLCGDict2
    implementations of Python bindings using the Reflection dictionaries.
    The current implementation is based on Reflex dictionaries with PyROOT.
    PyROOT provides the basic pyhton bindings to C++ and Cintex to populate 
    CINT dictionaries from Reflex ones.
"""
#---- system modules--------------------------------------------------
import os, sys, exceptions, string, warnings

#---- Import PyROOT module -------------------------------------------

## load PyROOT C++ extension module, special case for linux and Sun
needsGlobal =  ( 0 <= string.find( sys.platform, 'linux' ) ) or\
               ( 0 <= string.find( sys.platform, 'sunos' ) )
if needsGlobal:
 # change dl flags to load dictionaries from pre-linked .so's
   dlflags = sys.getdlopenflags()
   sys.setdlopenflags( 0x100 | 0x2 )    # RTLD_GLOBAL | RTLD_NOW

import libPyROOT

# reset dl flags if needed
if needsGlobal:
   sys.setdlopenflags( dlflags )
del needsGlobal

libPyROOT.SetMemoryPolicy( libPyROOT.kMemoryStrict )
libPyROOT.MakeRootClass( 'PyROOT::TPyROOTApplication' ).InitCINTMessageCallback()
#--- Enable Autoloading ignoring possible error for the time being
try:    libPyROOT.gInterpreter.EnableAutoLoading()
except: pass

#--- Load CINT dictionaries for STL classes first before other "Reflex" dictionaries
#    are loaded. The Reflex once are protected against the class being there 
#libPyROOT.gROOT.ProcessLine('int sav = gErrorIgnoreLevel; gErrorIgnoreLevel = 2001;')
#for c in ('vector', 'list', 'set', 'deque') : 
#  if libPyROOT.gSystem.Load('lib%sDict' % c ) == -1 :
#    warnings.warn('CINT dictionary for STL class %s could not be loaded' % c )
#libPyROOT.gROOT.ProcessLine('gErrorIgnoreLevel = sav;')

#--- template support ------------------------------------------------------------
class Template:
   def __init__( self, name ):
      self.__name__ = name
   def __call__( self, *args ):
      name = self.__name__[ 0 <= self.__name__.find( 'std::' ) and 5 or 0:]
      return libPyROOT.MakeRootTemplateClass( name, *args )

sys.modules[ 'libPyROOT' ].Template = Template


#--- LoadDictionary function and aliases -----------------------------
def loadDictionary(name) :
  # prepend "lib" 
  if sys.platform != 'win32' and name[:3] != 'lib' :
     name = 'lib' + name
  sc = libPyROOT.gSystem.Load(name)
  if sc == -1 : raise RuntimeError("Error Loading dictionary")
loadDict = loadDictionary

#--- Load Cintex module and enable conversions Reflex->CINT-----------
# TSystem::Load() already knows that cintex depends on reflex.
# libPyROOT.gSystem.Load('libReflex')
libPyROOT.gSystem.Load('libCintex')

Cintex = libPyROOT.MakeRootClass( 'Cintex' )
Cintex.SetDebug(0)
Cintex.Enable()

#--- Other functions needed -------------------------------------------
class _global_cpp: 
  class std:  #--- scope place holder for STL classes ------------------------------------------
     stlclasses = ( 'complex', 'exception', 'pair', 'deque', 'list', 'queue',\
                     'stack', 'vector', 'map', 'multimap', 'set', 'multiset' )
     for name in stlclasses:
        exec '%(name)s = Template( "std::%(name)s" )' % { 'name' : name }

libPyROOT.SetRootLazyLookup( _global_cpp.__dict__ ) 
 
def Namespace( name ) :
  if name == '' : return _global_cpp
  else :          return libPyROOT.LookupRootEntity(name)
makeNamespace = Namespace

def makeClass( name ) :
  return libPyROOT.MakeRootClass(name)
  
def addressOf( obj ) :
  return libPyROOT.AddressOf(obj)[0]
       
def getAllClasses( ) :
  TClassTable = makeClass('TClassTable')
  TClassTable.Init()
  classes = []
  while True :
    c = TClassTable.Next()
    if c : classes.append(c)
    else : break
  return classes

#--- Global namespace and global objects -------------------------------
gbl  = _global_cpp
NULL = 0 # libPyROOT.GetRootGlobal returns a descriptor, which needs a class
class double(float): pass
class short(int): pass
class long_int(int): pass
class unsigned_short(int): pass
class unsigned_int(int): pass
class unsigned_long(long): pass

#--- For test purposes --------------------------------------------------
if __name__ == '__main__' :
  loadDict('test_CintexDict')
