# @(#)root/pyroot:$Name:  $:$Id: ROOT.py,v 1.12 2004/10/30 06:26:43 brun Exp $
# Author: Wim Lavrijsen (WLavrijsen@lbl.gov)
# Created: 02/20/03
# Last: 11/01/04

"""Modify the exception hook to add ROOT classes as requested. Ideas stolen from
LazyPython (Nathaniel Gray <n8gray@caltech.edu>)."""

## system modules
import os, sys, exceptions, inspect, re
import string as pystring
import thread, time

## there's no version_info (nor inspect module) in 1.5.2
if sys.version[0:3] < '2.2':
    raise ImportError, 'Python Version 2.2 or above is required.'

## readline support, if available
try:
  import rlcompleter, readline
  readline.parse_and_bind( 'tab: complete' )
  readline.parse_and_bind( 'set show-all-if-ambiguous On' )
except:
  pass

## PyROOT C++ extension module
dlflags = sys.getdlopenflags()
if 0 <= pystring.find( sys.platform, 'linux' ):
   sys.setdlopenflags( 0x100 | 0x2 )    # RTLD_GLOBAL | RTLD_NOW
from libPyROOT import *
sys.setdlopenflags( dlflags )


## 2.2 has 10 instructions as default, 2.3 has 100 ... make same
sys.setcheckinterval( 100 )


### data ________________________________________________________________________
__version__ = '2.1.0'
__author__  = 'Wim Lavrijsen (WLavrijsen@lbl.gov)'

__all__ = [ 'makeRootClass', 'gROOT', 'gSystem', 'gInterpreter', 'gPad' ]

_NAME = 'name'
_NAMEREX = re.compile( r"named? '?(?P<%s>[\w\d]+)'?" % _NAME )

_orig_ehook = sys.excepthook


### helpers ---------------------------------------------------------------------
def split( str ):
   npos = pystring.find( str, ' ' )
   if 0 <= npos:
      return str[:npos], str[npos+1:]
   else:
      return str, ''


### special case for gPad (is a C++ macro) --------------------------------------
TVirtualPad = makeRootClass( "TVirtualPad" )

class _TVirtualPad( object ):
   def __getattribute__( self, what ):
      return getattr( TVirtualPad.Pad(), what )

   def __cmp__( self, other ):
      return cmp( TVirtualPad.Pad(), other )

   def __len__( self ):
      if TVirtualPad.Pad():
         return 1
      return 0

gPad = _TVirtualPad()


### exeption hook replacement ---------------------------------------------------
def _excepthook( exctype, value, traceb ):
 # catch name errors starting and fix if they are requests for ROOT classes
   if isinstance( value, exceptions.NameError ) or isinstance( value, exceptions.ImportError ):
      try:
         name = _NAMEREX.search( str(value) ).group( _NAME )
      except AttributeError:
         name = ''

      if not name:
         _orig_ehook( exctype, value, traceb )
         return

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
            try:
               exec '%s = makeRootClass( "%s" )' % (name,name) in glbdct, locdct
            except:
             # try global variables and global enums
               gGlobal = gROOT.GetGlobal( name, 1 )
               if not gGlobal:
                  gGlobal = getRootGlobalEnum( name )

               if gGlobal != None:
                  glbdct[ name ] = gGlobal
               else:
                  raise NameError
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

 # catch syntax errors to mimic ROOT commands
   elif isinstance( value, exceptions.SyntaxError ):
      cmd, arg = split( value.text[:-1] )

      if cmd == '.q':
         sys.exit( 0 )
      elif cmd == '.!':
         return os.system( arg )
      elif cmd == '.x':
         import __main__
         execfile( arg, __main__.__dict__, __main__.__dict__ )
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
    # allow "from ROOT import *" and name-completion
      self.__dict__[ '__all__' ] = __all__
      for name in __all__:
         exec 'self.__dict__[ "%s" ] = %s' % (name,name)

   def __getattr__( self, name ):
      if not hasattr( _thismodule, name ):
         try:
            exec 'global %s; %s = makeRootClass( "%s" )' % (name,name,name)
         except:
            aGlobal = gROOT.GetGlobal( name, 1 )
            if aGlobal == None:
               aGlobal = getRootGlobalEnum( name )

            if aGlobal:
               _thismodule.__dict__[ name ] = aGlobal
      return getattr( _thismodule, name )

sys.modules[ __name__ ] = ModuleFacade()
