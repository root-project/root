# @(#)root/pyroot:$Name:  $:$Id: ROOT.py,v 1.20 2005/04/13 05:04:49 brun Exp $
# Author: Wim Lavrijsen (WLavrijsen@lbl.gov)
# Created: 02/20/03
# Last: 05/04/05

"""PyROOT user module.

 o) install lazy ROOT class/variable lookup as appropriate
 o) feed gSystem and gInterpreter for display updates
 o) enable some ROOT/CINT style commands
 o) handle a few special cases such as gPad

"""

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

## load PyROOT C++ extension module, special case for linux and Sun
needsGlobal =  ( 0 <= pystring.find( sys.platform, 'linux' ) ) or\
               ( 0 <= pystring.find( sys.platform, 'sunos' ) )
if needsGlobal:
 # change dl flags to load dictionaries from pre-linked .so's
   dlflags = sys.getdlopenflags()
   sys.setdlopenflags( 0x100 | 0x2 )    # RTLD_GLOBAL | RTLD_NOW

from libPyROOT import *

# reset dl flags if needed
if needsGlobal:
   sys.setdlopenflags( dlflags )
del needsGlobal

## 2.2 has 10 instructions as default, 2.3 has 100 ... make same
sys.setcheckinterval( 100 )


### data ________________________________________________________________________
__version__ = '3.0.0'
__author__  = 'Wim Lavrijsen (WLavrijsen@lbl.gov)'

__pseudo__all__ = [ 'gROOT', 'gSystem', 'gInterpreter', 'gPad' ]

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


### RINT command emulation ------------------------------------------------------
def _excepthook( exctype, value, traceb ):
 # catch syntax errors only (they contain the full line)
   if isinstance( value, exceptions.SyntaxError ) and value.text:
      cmd, arg = split( value.text[:-1] )

    # mimic ROOT/CINT commands
      if cmd == '.q':
         sys.exit( 0 )
      elif cmd == '.!' and arg:
         return os.system( arg )
      elif cmd == '.x' and arg:
         import __main__
         fn = os.path.expanduser( os.path.expandvars( arg ) )
         execfile( fn, __main__.__dict__, __main__.__dict__ )
         return
      elif cmd == '.cd' and arg:
         os.chdir( arg )
         return
      elif cmd == '.ls':
         return sys.modules[ __name__ ].gDirectory.ls()
      elif cmd == '.pwd':
         return sys.modules[ __name__ ].gDirectory.pwd()

 # normal exception processing
   _orig_ehook( exctype, value, traceb )

if not __builtins__.has_key( '__IPYTHON__' ):
 # IPython has its own ways of executing shell commands etc.
   sys.excepthook = _excepthook


### call EndOfLineAction after each interactive command (to update display etc.)
def _displayhook( v ):
   gInterpreter.EndOfLineAction()
   return _orig_dhook( v )

_orig_dhook = sys.displayhook
sys.displayhook = _displayhook


### root thread -----------------------------------------------------------------
def _processRootEvents():
   global gSystem
   gSystemProcessEvents = gSystem.ProcessEvents
   while 1:
      try:
         gSystemProcessEvents()
         time.sleep( 0.01 )
      except: # in case gSystem gets destroyed early on exit
         pass

thread.start_new_thread( _processRootEvents, () )


### allow loading ROOT classes as attributes ------------------------------------
_thismodule = sys.modules[ __name__ ]

class ModuleFacade:
   def __init__( self ):
    # store already available ROOT objects to prevent spurious lookups
      for name in _thismodule.__pseudo__all__:
          self.__dict__[ name ] = getattr( _thismodule, name )

      self.__dict__[ '__doc__'  ] = _thismodule.__doc__
      self.__dict__[ '__name__' ] = _thismodule.__name__

   def __getattr__( self, name ):
    # support for "from ROOT import *" at the module level
      if name == '__all__':
         caller = sys.modules[ sys._getframe( 1 ).f_globals[ '__name__' ] ]

         for name in _thismodule.__pseudo__all__:
            caller.__dict__[ name ] = getattr( _thismodule, name )

         sys.modules[ 'libPyROOT' ].gPad = gPad

       # make the distionary of the calling module ROOT lazy
         _thismodule.setRootLazyLookup( caller.__dict__ )

       # pretend it was a failure to prevent further copying
         raise AttributeError( name )

    # block search for privates
      if name[0:2] == '__':
         raise AttributeError( name )

      try:
       # attempt to construct "name" as a ROOT class
         attr = makeRootClass( name )
      except:
       # no such class ... try global variable or global enum
         attr = getRootGlobal( name )

    # cache value locally so that we don't come back here
      if attr != None:
         self.__dict__[ name ] = attr
      else:
         raise AttributeError( name )

    # success!
      return attr

sys.modules[ __name__ ] = ModuleFacade()
