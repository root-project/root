from __future__ import generators
# @(#)root/pyroot:$Name:  $:$Id: ROOT.py,v 1.40 2006/05/29 15:54:05 brun Exp $
# Author: Wim Lavrijsen (WLavrijsen@lbl.gov)
# Created: 02/20/03
# Last: 06/12/06

"""PyROOT user module.

 o) install lazy ROOT class/variable lookup as appropriate
 o) feed gSystem and gInterpreter for display updates
 o) add readline completion (if supported by python build)
 o) enable some ROOT/CINT style commands
 o) handle a few special cases such as gPad, STL, etc.

"""

## system modules
import os, sys, time
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
if hasattr(sys,'argv') and '-b' in sys.argv and os.environ.has_key( 'DISPLAY' ):
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

## choose interactive-favoured policies
SetMemoryPolicy( kMemoryHeuristics )
SetSignalPolicy( kSignalSafe )

## normally, you'll want a ROOT application; don't init any further if
## one pre-exists from some C++ code somewhere
c = MakeRootClass( 'PyROOT::TPyROOTApplication' )
if c.CreatePyROOTApplication():
   c.InitROOTGlobals()
   c.InitCINTMessageCallback();
del c

## 2.2 has 10 instructions as default, 2.3 has 100 ... make same
sys.setcheckinterval( 100 )


### data ________________________________________________________________________
__version__ = '4.2.0'
__author__  = 'Wim Lavrijsen (WLavrijsen@lbl.gov)'

__pseudo__all__ = [ 'gROOT', 'gSystem', 'gInterpreter', 'gPad', 'gVirtualX',
                    'AddressOf', 'MakeNullPointer', 'Template', 'std' ]
__all__         = []                         # purposedly empty

_orig_ehook = sys.excepthook

## for setting memory and speed policies; not exported
_memPolicyAPI = [ 'SetMemoryPolicy', 'SetOwnership', 'kMemoryHeuristics', 'kMemoryStrict' ]
_sigPolicyAPI = [ 'SetSignalPolicy', 'kSignalFast', 'kSignalSafe' ]


### helpers ---------------------------------------------------------------------
def split( str ):
   npos = pystring.find( str, ' ' )
   if 0 <= npos:
      return str[:npos], str[npos+1:]
   else:
      return str, ''


### template support ------------------------------------------------------------
class Template:
   def __init__( self, name ):
      self.__name__ = name

   def __call__( self, *args ):
      newargs = [ self.__name__[ 0 <= self.__name__.find( 'std::' ) and 5 or 0:] ]
      for arg in args:
         if type(arg) == str:
            arg = pystring.join(
               map( lambda x: pystring.strip(x), pystring.split(arg,',') ), ',' )
         newargs.append( arg )
      return MakeRootTemplateClass( *newargs )

sys.modules[ 'libPyROOT' ].Template = Template


### scope place holder for STL classes ------------------------------------------
class std:
   stlclasses = ( 'complex', 'exception', 'pair', \
      'deque', 'list', 'queue', 'stack', 'vector', 'map', 'multimap', 'set', 'multiset' )

   for name in stlclasses:
      exec '%(name)s = Template( "std::%(name)s" )' % { 'name' : name }

sys.modules[ 'libPyROOT' ].std = std


### special cases for gPad, gVirtualX (are C++ macro's) -------------------------
class _ExpandMacroFunction( object ):
   def __init__( self, klass, func ):
      c = MakeRootClass( klass )
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


### special case pythonization --------------------------------------------------
def _TTree__iter__( self ):
  n = self.GetEntries()
  i = 0
  while i < n:
    self.GetEntry( i )
    yield self                  # TODO: not sure how to do this w/ C-API ...
    i += 1

MakeRootClass( "TTree" ).__iter__    = _TTree__iter__


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
   elif isinstance( value, SyntaxError ) and \
      value.msg == "can't assign to function call":
         print """Are you trying to assign a value to a reference return, for example to the
result of a call to "double& SMatrix<>::operator()(int,int)"? If so, then
please use operator[] instead, as in e.g. "mymatrix[i,j] = somevalue".
"""

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
      self.__dict__[ 'module' ]    = module
      self.__dict__[ 'libmodule' ] = sys.modules[ 'libPyROOT' ]

    # store already available ROOT objects to prevent spurious lookups
      for name in self.module.__pseudo__all__ + _memPolicyAPI + _sigPolicyAPI:
          self.__dict__[ name ] = getattr( self.module, name )

      for name in std.stlclasses:
          exec 'self.%(name)s = std.%(name)s' % { 'name' : name }

      self.__dict__[ '__doc__'  ] = self.module.__doc__
      self.__dict__[ '__name__' ] = self.module.__name__

      self.__dict__[ 'keeppolling' ] = 0

      self.__class__.__getattr__ = self.__class__.__getattr1

   def __setattr__( self, name, value ):
    # to allow assignments to ROOT globals such as ROOT.gDebug
      if not name in self.__dict__:
         try:
          # assignment to an existing ROOT global
            setattr( self.__class__, name, GetRootGlobal( name ) )
         except LookupError:
          # allow a few limited cases where new globals can be set
            tcnv = { int         : 'int %s = %d;',
                     long        : 'long %s = %d;',
                     float       : 'double %s = %f;',
                     str         : 'string %s = "%s";' }
            try:
               gROOT.ProcessLine( tcnv[ type(value) ] % (name,value) );
               setattr( self.__class__, name, GetRootGlobal( name ) )
            except KeyError:
               pass

      super( self.__class__, self ).__setattr__( name, value )

   def __getattr1( self, name ):
    # this is the "start-up" getattr, which handles the special cases

      if name == '__all__':
       # support for "from ROOT import *" at the module level
         caller = sys.modules[ sys._getframe( 1 ).f_globals[ '__name__' ] ]

         for name in self.module.__pseudo__all__:
            caller.__dict__[ name ] = getattr( self.module, name )

         self.libmodule.gPad = gPad

       # make the dictionary of the calling module ROOT lazy
         self.module.SetRootLazyLookup( caller.__dict__ )

       # all bets are off with import *, so follow -b flag
         self.__doGUIThread()

       # done with this version of __getattr__, move to general one
         self.__class__.__getattr__ = self.__class__.__getattr2

       # the actual __all__ is empty
         self.__dict__[ '__all__' ] = self.module.__all__
         return self.module.__all__

      elif name == 'gROOT':
       # yield gROOT without starting GUI thread just yet
         return self.module.gROOT

      elif name[0] != '_':
       # first request for non-private (i.e. presumable ROOT) entity
         self.__doGUIThread()

       # done with this version of __getattr__, move to general one
         self.__class__.__getattr__ = self.__class__.__getattr2
         return getattr( self, name )

   def __getattr2( self, name ):
    # this is the "running" getattr, which is simpler

    # lookup into ROOT (which may cause python-side enum/class/global creation)
      attr = self.libmodule.LookupRootEntity( name )

    # the call above will raise AttributeError as necessary; so if we get here,
    # attr is valid: cache as appropriate, so we don't come back
      if type(attr) == PropertyProxy:
          setattr( self.__class__, name, attr )        # descriptor
          return getattr( self, name )
      else:
          self.__dict__[ name ] = attr                 # normal member
          return attr

    # reaching this point means failure ...
      raise AttributeError( name )

   def __doGUIThread( self ):
    # root thread to prevent GUIs from starving, as needed
      if not self.keeppolling and not self.module.gROOT.IsBatch():
         import threading
         self.__dict__[ 'keeppolling' ] = 1
         self.__dict__[ 'thread' ] = \
            threading.Thread( None, _processRootEvents, None, ( self, ) )
         self.thread.setDaemon( 1 )
         self.thread.start()

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
      facade.thread.join( 3. )                         # arbitrary

 # destroy ROOT module
   del facade.libmodule
   del sys.modules[ 'libPyROOT' ]
   del facade.module

 # destroy facade
   del sys.modules[ facade.__name__ ]
   del facade

atexit.register( cleanup )
del cleanup, atexit
