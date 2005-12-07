# @(#)root/pyroot:$Name:  $:$Id: ROOT.py,v 1.33 2005/11/24 17:08:36 rdm Exp $
# Author: Wim Lavrijsen (WLavrijsen@lbl.gov)
# Created: 02/20/03
# Last: 12/06/05

"""PyROOT user module.

 o) install lazy ROOT class/variable lookup as appropriate
 o) feed gSystem and gInterpreter for display updates
 o) add readline completion (if supported by python build)
 o) enable some ROOT/CINT style commands
 o) handle a few special cases such as gPad, STL, etc.

"""

## system modules
import os, sys
import string as pystring

## there's no version_info in 1.5.2
if sys.version[0:3] < '2.2':
    raise ImportError, 'Python Version 2.2 or above is required.'

## readline support, if available
try:
   import rlcompleter, readline

   class FileNameCompleter( rlcompleter.Completer ):
      def file_matches( self, text ):
         matches = []
         path, name = os.path.split( text )

         try:
            for fn in os.listdir( path or os.curdir ):
               if fn[:len(name)] == name:
                  full = os.path.join( path, fn )
                  matches.append( full )

                  if os.path.isdir( full ):
                     matches += map( lambda x: os.path.join( full, x ), os.listdir( full ) )
         except OSError:
            pass

         return matches

      def global_matches( self, text ):
         matches = rlcompleter.Completer.global_matches( self, text )
         if not matches:
            matches = []
         return matches + self.file_matches( text )

   readline.set_completer( FileNameCompleter().complete )
   readline.set_completer_delims(
      pystring.replace( readline.get_completer_delims(), os.sep , '' ) )

   readline.parse_and_bind( 'tab: complete' )
   readline.parse_and_bind( 'set show-all-if-ambiguous On' )
except:
 # module readline typically doesn't exist on non-Unix platforms
   pass

## remove DISPLAY variable in batch mode as not confuse early ROOT calls
if '-b' in sys.argv and os.environ.has_key( 'DISPLAY' ):
    del os.environ[ 'DISPLAY' ]       

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

## normally, you'll want a ROOT application; fine if one pre-exists from C++
InitRootApplication()

## 2.2 has 10 instructions as default, 2.3 has 100 ... make same
sys.setcheckinterval( 100 )


### data ________________________________________________________________________
__version__ = '3.3.0'
__author__  = 'Wim Lavrijsen (WLavrijsen@lbl.gov)'

__pseudo__all__ = [ 'gROOT', 'gSystem', 'gInterpreter', 'gPad', 'gVirtualX',
                    'AddressOf', 'MakeNullPointer', 'Template', 'std' ]
__all__         = []                         # purposedly empty

_orig_ehook = sys.excepthook

## for setting memory policies; not exported
_memPolicyAPI = [ 'SetMemoryPolicy', 'SetOwnership', 'kMemoryHeuristics', 'kMemoryStrict' ]
kMemoryHeuristics = 1
kMemoryStrict     = 2

## speed hack
_sigPolicyAPI = [ 'SetSignalPolicy', 'kSignalFast', 'kSignalSafe' ]
kSignalFast = 1
kSignalSafe = 2


### helpers ---------------------------------------------------------------------
def split( str ):
   npos = pystring.find( str, ' ' )
   if 0 <= npos:
      return str[:npos], str[npos+1:]
   else:
      return str, ''

def safeLookupCall( func, arg ):
   try:
      return func( arg )
   except:
      return None


### template support ------------------------------------------------------------
class Template:
   def __init__( self, name ):
      self.__name__ = name

   def __call__( self, *args ):
      name = self.__name__[ 0 <= self.__name__.find( 'std::' ) and 5 or 0:]
      return MakeRootTemplateClass( name, *args )

sys.modules[ 'libPyROOT' ].Template = Template


### scope place holder for STL classes ------------------------------------------
class std:
   stlclasses = ( 'complex', 'exception', 'pair', \
      'deque', 'list', 'queue', 'stack', 'vector', 'map', 'multimap', 'set', 'multiset' )

   for name in stlclasses:
      exec '%(name)s = Template( "std::%(name)s" )' % { 'name' : name }


### special cases for gPad, gVirtualX (are C++ macro's) -------------------------
class _ExpandMacroFunction( object ):
   def __init__( self, klass, func ):
      c = makeRootClass( klass )
      self.func = getattr( c, func )

   def __getattr__( self, what ):
      return getattr( self.__dict__[ 'func' ](), what )

   def __cmp__( self, other ):
      return cmp( self.func(), other )

   def __len__( self ):
      if self.func():
         return 1
      return 0

gPad      = _ExpandMacroFunction( "TVirtualPad", "Pad" )
gVirtualX = _ExpandMacroFunction( "TVirtualX",   "Instance" )


### RINT command emulation ------------------------------------------------------
def _excepthook( exctype, value, traceb ):
 # catch syntax errors only (they contain the full line)
   if isinstance( value, SyntaxError ) and value.text:
      cmd, arg = split( value.text[:-1] )

    # mimic ROOT/CINT commands
      if cmd == '.q':
         sys.exit( 0 )
      elif cmd == '.?' or cmd == '.help':
         print """PyROOT emulation of CINT commands.
All emulated commands must be preceded by a . (dot).
===========================================================================
Help:        ?         : this help
             help      : this help
Shell:       ![shell]  : execute shell command
Evaluation:  x [file]  : load [file] and evaluate {statements} in the file
Load/Unload: L [lib]   : load [lib]
Quit:        q         : quit python session

The standard python help system is available through a call to 'help()' or
'help(<id>)' where <id> is an identifier, e.g. a class or function such as
TPad or TPad.cd, etc."""
         return
      elif cmd == '.!' and arg:
         return os.system( arg )
      elif cmd == '.x' and arg:
         import __main__
         fn = os.path.expanduser( os.path.expandvars( arg ) )
         execfile( fn, __main__.__dict__, __main__.__dict__ )
         return
      elif cmd == '.L':
         return gSystem.Load( arg )
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
else:
 # IPython's FakeModule hack otherwise prevents usage of python from CINT
   gROOT.ProcessLine( 'TPython::Exec( "" )' )
   sys.modules[ '__main__' ].__builtins__ = __builtins__


### call EndOfLineAction after each interactive command (to update display etc.)
def _displayhook( v ):
   gInterpreter.EndOfLineAction()
   return _orig_dhook( v )

_orig_dhook = sys.displayhook
sys.displayhook = _displayhook


### helper to prevent GUIs from starving
def _processRootEvents( controller ):
    import time
    global gSystem

    while controller.keeppolling:
       try:
          gSystem.ProcessEvents()
          time.sleep( 0.01 )
       except: # in case gSystem gets destroyed early on exit
          pass


### allow loading ROOT classes as attributes ------------------------------------
class ModuleFacade( object ):
   def __init__( self, module ):
      self.module = module

    # root thread to prevent GUIs from starving
      if not self.module.gROOT.IsBatch():
         import threading, time
         self.keeppolling = 1
         self.thread = threading.Thread( None, _processRootEvents, None, ( self, ) )
         self.thread.start()

    # store already available ROOT objects to prevent spurious lookups
      for name in self.module.__pseudo__all__ + _memPolicyAPI + _sigPolicyAPI:
          self.__dict__[ name ] = getattr( self.module, name )

      for name in std.stlclasses:
          exec 'self.%(name)s = std.%(name)s' % { 'name' : name }

      self.__dict__[ '__doc__'  ] = self.module.__doc__
      self.__dict__[ '__name__' ] = self.module.__name__

   def __getattr__( self, name ):
    # support for "from ROOT import *" at the module level
      if name == '__all__':
         caller = sys.modules[ sys._getframe( 1 ).f_globals[ '__name__' ] ]

         for name in self.module.__pseudo__all__:
            caller.__dict__[ name ] = getattr( self.module, name )

         sys.modules[ 'libPyROOT' ].gPad = gPad

       # make the distionary of the calling module ROOT lazy
         self.module.setRootLazyLookup( caller.__dict__ )

       # the actual __all__ is empty
         return self.module.__all__

    # block search for privates
      if name[0:2] == '__':
         raise AttributeError( name )

    # attempt to construct "name" as a ROOT class
      attr = safeLookupCall( makeRootClass, name )
      if ( attr == None ):
       # no such class ... try global variable or global enum
         attr = safeLookupCall( getRootGlobal, name )
      
      if ( attr == None ):
       # no global either ... try through gROOT (e.g. objects from files)
         attr = gROOT.FindObject( name )

    # if available, cache attribute as appropriate, so we don't come back
      if attr != None:
         if type(attr) == PropertyProxy:
            setattr( self.__class__, name, attr )      # descriptor
            return getattr( self, name )
         else:
            self.__dict__[ name ] = attr               # normal member
            return attr

    # reaching this point means failure ...
      raise AttributeError( name )

sys.modules[ __name__ ] = ModuleFacade( sys.modules[ __name__ ] )
del ModuleFacade


### b/c of circular references, the facade needs explicit cleanup ---------------
import atexit
def cleanup():
 # restore hooks
   import sys
   sys.displayhook = sys.__displayhook__
   if not __builtins__.has_key( '__IPYTHON__' ):
      sys.excepthook = sys.__excepthook__

   facade = sys.modules[ __name__ ]

 # shutdown GUI thread, as appropriate
   if hasattr( facade, 'thread' ):
      facade.keeppolling = 0

 # destroy ROOT module
   del sys.modules[ 'libPyROOT' ]
   del facade.module

 # destroy facade
   del sys.modules[ facade.__name__ ]
   del facade

atexit.register( cleanup )
del cleanup, atexit
