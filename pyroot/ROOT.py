# @(#)root/pyroot:$Name:  $:$Id: ROOT.py,v 1.3 2004/04/28 22:04:06 rdm Exp $
# Author: Wim Lavrijsen (WLavrijsen@lbl.gov)
# Created: 02/20/03
# Last: 04/29/04

"""Modify the exception hook to add ROOT classes as requested. Ideas stolen from
LazyPython (Nathaniel Gray <n8gray@caltech.edu>)."""

## system modules
import sys, exceptions, inspect, string, re
import thread, time

## There's no version_info (nor inspect module) in 1.5.2.
if sys.version[0:3] < '2.1':
    raise ImportError, 'Python Version 2.1 or above is required.'

## readline support
try:
  import rlcompleter, readline
  readline.parse_and_bind( 'tab: complete' )
  readline.parse_and_bind( 'set show-all-if-ambiguous On' )
except:
  pass

## PyROOT C++ extension module
from libPyROOT import * 

### load most common ROOT libraries______________________________________________
libraries = ('libHist', 'libGpad', 'libGraf', 'libMatrix', 'libTree', 
             'libGraf3d', 'libGeom' )
for l in libraries : gSystem.Load(l)

### data ________________________________________________________________________
_NAME = 'name'
_NAMEREX = re.compile( r"named? '?(?P<%s>[\w\d]+)'?" % _NAME )

_orig_ehook = sys.excepthook

kWhite, kBlack, kRed, kGreen, kBlue, kYellow, kMagenta, kCyan = range( 0, 8 )

gGeometry = gROOT.GetGlobal( 'gGeometry' )


### exeption hook replacement ---------------------------------------------------
def _excepthook( exctype, value, traceb ):
   name = ''

 # catch name errors starting and fix if they are requests for ROOT classes
   if isinstance( value, exceptions.NameError ) or isinstance( value, exceptions.ImportError ):
      res = _NAMEREX.search( str(value) )
      if res:
         name = res.group( _NAME )

   if name:
    # get last frame from the exception stack (for its dictionaries and code)
      lfr = inspect.getinnerframes( traceb )[-1][0]

      if isinstance( value, exceptions.NameError ):
         glbdct, locdct = lfr.f_globals, lfr.f_locals
      else:
         glbdct = sys.modules[ __name__ ].__dict__
         locdct = glbdct

     # attempt to construct a ROOT class ...
      if glbdct.has_key( 'makeRootClass' ) or locdct.has_key( 'makeRootClass' ):
         try:
          # construct the ROOT shadow class in the current and the ROOT module (may throw)
            exec '%s = makeRootClass( "%s" )' % (name,name) in glbdct, locdct
         except (NameError, TypeError):      # ROOT not loaded or class lookup failed
            _orig_ehook( exctype, value, traceb )
         else:
            try:                             # ok, once again (may still fail)
               exec lfr.f_code in lfr.f_globals, lfr.f_locals
            except (NameError, ImportError): # recurse
               info = sys.exc_info()[:2]
               _excepthook( info[0], info[1], traceb )
            except:                          # new error, report on old traceback
               info = sys.exc_info()[:2]
               _orig_ehook( info[0], info[1], traceb )
      else:
         _orig_ehook( exctype, value, traceb )

      del lfr                                # no cycles, please ...
      return

 # normal exception processing
   _orig_ehook( exctype, value, traceb )


if __builtins__.has_key( '__IPYTHON__' ):
 # ouch! :P the following is going to be tricky ...
   _orig_stback = __IPYTHON__.showtraceback

   def _showtraceback( exc_tuple = None ):
      if not exc_tuple:
         type, value, tb = sys.exc_info()
      else:
         type, value, tb = exc_tuple

      _excepthook( type, value, tb )

   def _call_orig_stback( type, value, tb ):
      _orig_stback( (type,value,tb) )

   _orig_ehook = _call_orig_stback
   __IPYTHON__.showtraceback = _showtraceback
else:
   sys.excepthook = _excepthook


### call EndOfLineAction after each interactive command
def _displayhook( v ):
   gInterpreter.EndOfLineAction()
   return _orig_dhook( v )

_orig_dhook = sys.displayhook
sys.displayhook = _displayhook


### root thread -----------------------------------------------------------------
def _processRootEvents():
   global gSystem
   while 1:
      gSystem.ProcessEvents()
      time.sleep( 0.01 )

thread.start_new_thread( _processRootEvents, () )


### allow loading ROOT classes as attributes ------------------------------------
_thismodule = sys.modules[ __name__ ]

class ModuleFacade:
   def __init__( self ):
    # allow "from ROOT import *"
      self.__dict__[ 'makeRootClass' ] = makeRootClass
      self.__dict__[ '__all__' ] = [ 'makeRootClass' ]

   def __getattr__( self, name ):
      if not hasattr( _thismodule, name ):
         exec 'global %s; %s = makeRootClass( "%s" )' % (name,name,name)
      return getattr( _thismodule, name )

sys.modules[ __name__ ] = ModuleFacade()
