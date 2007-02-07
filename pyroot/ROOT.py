from __future__ import generators
# @(#)root/pyroot:$Name:  $:$Id: ROOT.py,v 1.45 2006/12/08 07:42:31 brun Exp $
# Author: Wim Lavrijsen (WLavrijsen@lbl.gov)
# Created: 02/20/03
# Last: 02/06/07

"""PyROOT user module.

 o) install lazy ROOT class/variable lookup as appropriate
 o) feed gSystem and gInterpreter for display updates
 o) add readline completion (if supported by python build)
 o) enable some ROOT/CINT style commands
 o) handle a few special cases such as gPad, STL, etc.

"""

__version__ = '5.0.0'
__author__  = 'Wim Lavrijsen (WLavrijsen@lbl.gov)'


### system and interpreter setup ------------------------------------------------
import os, sys, time
import string as pystring

## there's no version_info in 1.5.2
if sys.version[0:3] < '2.2':
    raise ImportError, 'Python Version 2.2 or above is required.'

## 2.2 has 10 instructions as default, > 2.3 has 100 ... make same
sys.setcheckinterval( 100 )

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

## remove DISPLAY variable in batch mode as to not confuse early ROOT calls
if hasattr(sys,'argv') and '-b' in sys.argv and os.environ.has_key( 'DISPLAY' ):
   del os.environ[ 'DISPLAY' ]       


### load PyROOT C++ extension module, special case for linux and Sun ------------
needsGlobal =  ( 0 <= pystring.find( sys.platform, 'linux' ) ) or\
               ( 0 <= pystring.find( sys.platform, 'sunos' ) )
if needsGlobal:
 # change dl flags to load dictionaries from pre-linked .so's
   dlflags = sys.getdlopenflags()
   sys.setdlopenflags( 0x100 | 0x2 )    # RTLD_GLOBAL | RTLD_NOW

import libPyROOT as _root

# reset dl flags if needed
if needsGlobal:
   sys.setdlopenflags( dlflags )
del needsGlobal


### configuration ---------------------------------------------------------------
class _Configuration( object ):
   __slots__ = [ 'StartGuiThread' ]

   def __init__( self ):
      self.StartGuiThread = 1

PyConfig = _Configuration()
del _Configuration


### choose interactive-favored policies -----------------------------------------
_root.SetMemoryPolicy( _root.kMemoryHeuristics )
_root.SetSignalPolicy( _root.kSignalSafe )


### data ________________________________________________________________________
__pseudo__all__ = [ 'gROOT', 'gSystem', 'gInterpreter',
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
      return _root.MakeRootTemplateClass( *newargs )

_root.Template = Template


### scope place holder for STL classes ------------------------------------------
class std:
   stlclasses = ( 'complex', 'exception', 'pair', \
      'deque', 'list', 'queue', 'stack', 'vector', 'map', 'multimap', 'set', 'multiset' )

   for name in stlclasses:
      exec '%(name)s = Template( "std::%(name)s" )' % { 'name' : name }

   string = _root.MakeRootClass( 'string' )

_root.std = std


### special cases for gPad, gVirtualX (are C++ macro's) -------------------------
class _ExpandMacroFunction( object ):
   def __init__( self, klass, func ):
      c = _root.MakeRootClass( klass )
      self.func = getattr( c, func )

   def __getattr__( self, what ):
      return getattr( self.__dict__[ 'func' ](), what )

   def __cmp__( self, other ):
      return cmp( self.func(), other )

   def __len__( self ):
      if self.func():
         return 1
      return 0

_root.gPad      = _ExpandMacroFunction( "TVirtualPad", "Pad" )
_root.gVirtualX = _ExpandMacroFunction( "TVirtualX",   "Instance" )


### special case pythonization --------------------------------------------------
def _TTree__iter__( self ):
   n = self.GetEntries()
   i = 0
   while i < n:
      self.GetEntry(i)
      yield self                   # TODO: not sure how to do this w/ C-API ...
      i += 1

_root.MakeRootClass( "TTree" ).__iter__    = _TTree__iter__


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
         return _root.gSystem.Load( arg )
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


### call EndOfLineAction after each interactive command (to update display etc.)
def _displayhook( v ):
   _root.gInterpreter.EndOfLineAction()
   return _orig_dhook( v )

_orig_dhook = sys.displayhook
sys.displayhook = _displayhook


### helper to prevent GUIs from starving
def _processRootEvents( controller ):
   gSystem = _root.gSystem

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

      self.__dict__[ '__doc__'  ] = self.module.__doc__
      self.__dict__[ '__name__' ] = self.module.__name__

      self.__dict__[ 'keeppolling' ] = 0
      self.__dict__[ 'PyConfig' ]    = self.module.PyConfig

      class gROOTWrapper( object ):
         def __init__( self, gROOT, master ):
            self.__dict__[ '_gROOT' ]  = gROOT
            self.__dict__[ '_master' ] = master

         def __getattr__( self, name ):
           if name != 'SetBatch':
              self._master._ModuleFacade__finalSetup()
              self._master.__dict__[ 'gROOT' ] = self._gROOT
           return getattr( self._gROOT, name )

         def __setattr__( self, name, value ):
           return setattr( self._gROOT, name, value )
              
      self.__dict__[ 'gROOT' ] = gROOTWrapper( _root.gROOT, self )

    # begin with startup gettattr/setattr
      self.__class__.__getattr__ = self.__class__.__getattr1
      self.__class__.__setattr__ = self.__class__.__setattr1

   def __setattr1( self, name, value ):      # "start-up" setattr
    # switch to running gettattr/setattr
      self.__class__.__getattr__ = self.__class__.__getattr2
      self.__class__.__setattr__ = self.__class__.__setattr2

    # create application, thread etc.
      self.__finalSetup()

    # let "running" setattr handle setting
      return setattr( self, name, value )

   def __setattr2( self, name, value ):     # "running" getattr
    # to allow assignments to ROOT globals such as ROOT.gDebug
      if not name in self.__dict__:
         try:
          # assignment to an existing ROOT global (establishes proxy)
            setattr( self.__class__, name, _root.GetRootGlobal( name ) )
         except LookupError:
          # allow a few limited cases where new globals can be set
            tcnv = { bool        : 'bool %s = %d;',
                     int         : 'int %s = %d;',
                     long        : 'long %s = %d;',
                     float       : 'double %s = %f;',
                     str         : 'string %s = "%s";' }
            try:
               _root.gROOT.ProcessLine( tcnv[ type(value) ] % (name,value) );
               setattr( self.__class__, name, _root.GetRootGlobal( name ) )
            except KeyError:
               pass           # can still assign normally, to the module

    # actual assignment through descriptor, or normal python way
      super( self.__class__, self ).__setattr__( name, value )

   def __getattr1( self, name ):             # "start-up" getattr
    # special case, to allow "from ROOT import gROOT" w/o starting GUI thread
      if name == '__path__':
         raise AttributeError( name )

    # switch to running gettattr/setattr
      self.__class__.__getattr__ = self.__class__.__getattr2
      self.__class__.__setattr__ = self.__class__.__setattr2

    # create application, thread etc.
      self.__finalSetup()

    # let "running" getattr handle lookup
      return getattr( self, name )

   def __getattr2( self, name ):             # "running" getattr
    # handle "from ROOT import *" ... can be called multiple times
      if name == '__all__':
         caller = sys.modules[ sys._getframe( 1 ).f_globals[ '__name__' ] ]

       # we may be calling in from __getattr1, verify and if so, go one frame up
         if caller == self:
            caller = sys.modules[ sys._getframe( 2 ).f_globals[ '__name__' ] ]

       # setup the pre-defined globals
         for name in self.module.__pseudo__all__:
            caller.__dict__[ name ] = getattr( _root, name )

       # install the hook
         _root.SetRootLazyLookup( caller.__dict__ )

       # return empty list, to prevent further copying
         return self.module.__all__

    # lookup into ROOT (which may cause python-side enum/class/global creation)
      attr = _root.LookupRootEntity( name )

    # the call above will raise AttributeError as necessary; so if we get here,
    # attr is valid: cache as appropriate, so we don't come back
      if type(attr) == _root.PropertyProxy:
         setattr( self.__class__, name, attr )         # descriptor
         return getattr( self, name )
      else:
         self.__dict__[ name ] = attr                  # normal member
         return attr

    # reaching this point means failure ...
      raise AttributeError( name )

   def __finalSetup( self ):
    # normally, you'll want a ROOT application; don't init any further if
    # one pre-exists from some C++ code somewhere
      c = _root.MakeRootClass( 'PyROOT::TPyROOTApplication' )
      if c.CreatePyROOTApplication():
         c.InitROOTGlobals()
         c.InitCINTMessageCallback();

    # must be called after gApplication creation:
      if __builtins__.has_key( '__IPYTHON__' ):
       # IPython's FakeModule hack otherwise prevents usage of python from CINT
         _root.gROOT.ProcessLine( 'TPython::Exec( "" )' )
         sys.modules[ '__main__' ].__builtins__ = __builtins__

    # root thread, if needed, to prevent GUIs from starving, as needed
      if self.PyConfig.StartGuiThread and \
            not ( self.keeppolling or _root.gROOT.IsBatch() ):
         import threading
         self.__dict__[ 'keeppolling' ] = 1
         self.__dict__[ 'thread' ] = \
            threading.Thread( None, _processRootEvents, None, ( self, ) )
         self.thread.setDaemon( 1 )
         self.thread.start()

    # store already available ROOT objects to prevent spurious lookups
      for name in self.module.__pseudo__all__ + _memPolicyAPI + _sigPolicyAPI:
         self.__dict__[ name ] = getattr( _root, name )

      for name in std.stlclasses:
         setattr( _root, name, getattr( std, name ) )


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

 # prevent spurious lookups into ROOT libraries
   del facade.__class__.__getattr__
   del facade.__class__.__setattr__

 # shutdown GUI thread, as appropriate
   if hasattr( facade, 'thread' ):
      facade.keeppolling = 0
      facade.thread.join( 3. )                         # arbitrary

 # destroy ROOT module
   del facade.module._root
   del sys.modules[ 'libPyROOT' ]
   del facade.module

 # destroy facade
   del sys.modules[ facade.__name__ ]
   del facade

atexit.register( cleanup )
del cleanup, atexit
