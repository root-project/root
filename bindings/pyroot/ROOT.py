from __future__ import generators
# @(#)root/pyroot:$Id$
# Author: Wim Lavrijsen (WLavrijsen@lbl.gov)
# Created: 02/20/03
# Last: 03/25/15

"""PyROOT user module.

 o) install lazy ROOT class/variable lookup as appropriate
 o) feed gSystem and gInterpreter for display updates
 o) add readline completion (if supported by python build)
 o) enable some ROOT/Cling style commands
 o) handle a few special cases such as gPad, STL, etc.
 o) execute rootlogon.py/.C scripts

"""

__version__ = '8.0.0'
__author__  = 'Wim Lavrijsen (WLavrijsen@lbl.gov)'


### system and interpreter setup ------------------------------------------------
import os, sys, types
import cppyy

## there's no version_info in 1.5.2
if sys.version[0:3] < '2.2':
   raise ImportError( 'Python Version 2.2 or above is required.' )

## 2.2 has 10 instructions as default, > 2.3 has 100 ... make same
if sys.version[0:3] == '2.2':
   sys.setcheckinterval( 100 )

## readline support, if available
try:
   import rlcompleter, readline

   class RootNameCompleter( rlcompleter.Completer ):
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

      def root_global_matches( self, text, prefix = '' ):
         gClassTable = _root.GetCppGlobal( 'gClassTable' )
         all = [ gClassTable.At(i) for i in xrange(gClassTable.Classes()) ]
         all += [ g.GetName() for g in _root.gROOT.GetListOfGlobals() ]
         matches = filter( lambda x: x[:len(text)] == text, all )
         return [prefix + x for x in matches]

      def global_matches( self, text ):
         matches = rlcompleter.Completer.global_matches( self, text )
         if not matches: matches = []
         matches += self.file_matches( text )
         return matches

      def attr_matches( self, text ):
         matches = rlcompleter.Completer.attr_matches( self, text )
         if not matches: matches = []
         b = text.find('.')
         try:
            if 0 <= b and self.namespace[text[:b]].__name__ == 'ROOT':
               matches += self.root_global_matches( text[b+1:], text[:b+1] )
         except AttributeError:    # not all objects have a __name__
            pass
         return matches

   readline.set_completer( RootNameCompleter().complete )
   readline.set_completer_delims(
      readline.get_completer_delims().replace( os.sep , '' ) )

   readline.parse_and_bind( 'tab: complete' )
   readline.parse_and_bind( 'set show-all-if-ambiguous On' )
except:
 # module readline typically doesn't exist on non-Unix platforms
   pass

## special filter on MacOS X (warnings caused by linking that is still required)
if sys.platform == 'darwin':
   import warnings
   warnings.filterwarnings( action='ignore', category=RuntimeWarning, module='ROOT',\
      message='class \S* already in TClassTable$' )

### load PyROOT C++ extension module, special case for linux and Sun ------------
_root = cppyy._backend

## convince 2.2 it's ok to use the expand function
if sys.version[0:3] == '2.2':
   import copy_reg
   copy_reg.constructor( _root._ObjectProxy__expand__ )

## convince inspect that PyROOT method proxies are possible drop-ins for python
## methods and classes for pydoc
import inspect

inspect._old_isfunction = inspect.isfunction
def isfunction( object ):
   if type(object) == _root.MethodProxy and not object.im_class:
      return True
   return inspect._old_isfunction( object )
inspect.isfunction = isfunction

inspect._old_ismethod = inspect.ismethod
def ismethod( object ):
   if type(object) == _root.MethodProxy:
      return True
   return inspect._old_ismethod( object )
inspect.ismethod = ismethod

del isfunction, ismethod


### configuration ---------------------------------------------------------------
class _Configuration( object ):
   __slots__ = [ 'IgnoreCommandLineOptions', 'StartGuiThread', 'ExposeCppMacros', '_gts' ]

   def __init__( self ):
      self.IgnoreCommandLineOptions = 0
      self.StartGuiThread = True
      self.ExposeCppMacros = False
      self._gts = []

   def __setGTS( self, value ):
      for c in value:
         if not callable( c ):
            raise ValueError( '"%s" is not callable' % str(c) );
      self._gts = value

   def __getGTS( self ):
      return self._gts

   GUIThreadScheduleOnce = property( __getGTS, __setGTS )

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
   npos = str.find( ' ' )
   if 0 <= npos:
      return str[:npos], str[npos+1:]
   else:
      return str, ''


### put std namespace directly onto ROOT ----------------------------------------
_root.std = cppyy.gbl.std
sys.modules['ROOT.std'] = cppyy.gbl.std


### special cases for gPad, gVirtualX (are C++ macro's) -------------------------
class _ExpandMacroFunction( object ):
   def __init__( self, klass, func ):
      c = _root.CreateScopeProxy( klass )
      self.func = getattr( c, func )

   def __getattr__( self, what ):
      return getattr( self.__dict__[ 'func' ](), what )

   def __cmp__( self, other ):
      return cmp( self.func(), other )

   def __nonzero__( self ):
      if self.func():
         return True
      return False

   def __repr__( self ):
      return repr( self.func() )

   def __str__( self ):
      return str( self.func() )

_root.gPad         = _ExpandMacroFunction( "TVirtualPad",  "Pad" )
_root.gVirtualX    = _ExpandMacroFunction( "TVirtualX",    "Instance" )
_root.gDirectory   = _ExpandMacroFunction( "TDirectory",   "CurrentDirectory" )
_root.gFile        = _ExpandMacroFunction( "TFile",        "CurrentFile" )
_root.gInterpreter = _ExpandMacroFunction( "TInterpreter", "Instance" )


### special case pythonization --------------------------------------------------
def _TTree__iter__( self ):
   i = 0
   bytes_read = self.GetEntry(i)
   while 0 < bytes_read:
      yield self                   # TODO: not sure how to do this w/ C-API ...
      i += 1
      bytes_read = self.GetEntry(i)

   if bytes_read == -1:
      raise RuntimeError( "TTree I/O error" )

_root.CreateScopeProxy( "TTree" ).__iter__    = _TTree__iter__


### RINT command emulation ------------------------------------------------------
def _excepthook( exctype, value, traceb ):
 # catch syntax errors only (they contain the full line)
   if isinstance( value, SyntaxError ) and value.text:
      cmd, arg = split( value.text[:-1] )

    # mimic ROOT/Cling commands
      if cmd == '.q':
         sys.exit( 0 )
      elif cmd == '.?' or cmd == '.help':
         sys.stdout.write( """PyROOT emulation of Cling commands.
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
TPad or TPad.cd, etc.
""" )
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
         sys.stdout.write( """Are you trying to assign a value to a reference return, for example to the
result of a call to "double& SMatrix<>::operator()(int,int)"? If so, then
please use operator[] instead, as in e.g. "mymatrix[i][j] = somevalue".
""" )

 # normal exception processing
   _orig_ehook( exctype, value, traceb )

if not '__IPYTHON__' in __builtins__:
 # IPython has its own ways of executing shell commands etc.
   sys.excepthook = _excepthook


### call EndOfLineAction after each interactive command (to update display etc.)
_orig_dhook = sys.displayhook
def _displayhook( v ):
   _root.gInterpreter.EndOfLineAction()
   return _orig_dhook( v )


### set import hook to be able to trigger auto-loading as appropriate
try:
   import __builtin__
except ImportError:
   import builtins as __builtin__  # name change in p3
_orig_ihook = __builtin__.__import__
def _importhook( name, *args, **kwds ):
   if name[0:5] == 'ROOT.':
      try:
         sys.modules[ name ] = getattr( sys.modules[ 'ROOT' ], name[5:] )
      except Exception:
         pass
   return _orig_ihook( name, *args, **kwds )

__builtin__.__import__ = _importhook


### helper to prevent GUIs from starving
def _processRootEvents( controller ):
   import time
   gSystemProcessEvents = _root.gSystem.ProcessEvents

   if sys.platform == 'win32':
      import thread
      _root.gROOT.ProcessLineSync('((TGWin32 *)gVirtualX)->SetUserThreadId(%ld)' % (thread.get_ident()))

   while controller.keeppolling:
      try:
         gSystemProcessEvents()
         if PyConfig.GUIThreadScheduleOnce:
            for guicall in PyConfig.GUIThreadScheduleOnce:
               guicall()
            PyConfig.GUIThreadScheduleOnce = []
         time.sleep( 0.01 )
      except: # in case gSystem gets destroyed early on exit
         pass


### allow loading ROOT classes as attributes ------------------------------------
class ModuleFacade( types.ModuleType ):
   def __init__( self, module ):
      types.ModuleType.__init__( self, 'ROOT' )

      self.__dict__[ 'module' ]   = module

      self.__dict__[ '__doc__'  ] = self.module.__doc__
      self.__dict__[ '__name__' ] = self.module.__name__
      self.__dict__[ '__file__' ] = self.module.__file__

      self.__dict__[ 'keeppolling' ] = 0
      self.__dict__[ 'PyConfig' ]    = self.module.PyConfig

      class gROOTWrapper( object ):
         def __init__( self, gROOT, master ):
            self.__dict__[ '_master' ] = master
            self.__dict__[ '_gROOT' ]  = gROOT

         def __getattr__( self, name ):
           if name != 'SetBatch' and self._master.__dict__[ 'gROOT' ] != self._gROOT:
              self._master._ModuleFacade__finalSetup()
              del self._master.__class__._ModuleFacade__finalSetup
           return getattr( self._gROOT, name )

         def __setattr__( self, name, value ):
           return setattr( self._gROOT, name, value )
              
      self.__dict__[ 'gROOT' ] = gROOTWrapper( _root.gROOT, self )
      del gROOTWrapper

    # begin with startup gettattr/setattr
      self.__class__.__getattr__ = self.__class__.__getattr1
      del self.__class__.__getattr1
      self.__class__.__setattr__ = self.__class__.__setattr1
      del self.__class__.__setattr1

   def __setattr1( self, name, value ):      # "start-up" setattr
    # create application, thread etc.
      self.__finalSetup()
      del self.__class__.__finalSetup

    # let "running" setattr handle setting
      return setattr( self, name, value )

   def __setattr2( self, name, value ):     # "running" getattr
    # to allow assignments to ROOT globals such as ROOT.gDebug
      if not name in self.__dict__:
         try:
          # assignment to an existing ROOT global (establishes proxy)
            setattr( self.__class__, name, _root.GetCppGlobal( name ) )
         except LookupError:
          # allow a few limited cases where new globals can be set
            if sys.hexversion >= 0x3000000:
               pylong = int
            else:
               pylong = long
            tcnv = { bool        : 'bool %s = %d;',
                     int         : 'int %s = %d;',
                     pylong      : 'long %s = %d;',
                     float       : 'double %s = %f;',
                     str         : 'string %s = "%s";' }
            try:
               _root.gROOT.ProcessLine( tcnv[ type(value) ] % (name,value) );
               setattr( self.__class__, name, _root.GetCppGlobal( name ) )
            except KeyError:
               pass           # can still assign normally, to the module

    # actual assignment through descriptor, or normal python way
      return super( self.__class__, self ).__setattr__( name, value )

   def __getattr1( self, name ):             # "start-up" getattr
    # special case, to allow "from ROOT import gROOT" w/o starting GUI thread
      if name == '__path__':
         raise AttributeError( name )

    # create application, thread etc.
      self.__finalSetup()
      del self.__class__.__finalSetup

    # let "running" getattr handle lookup
      return getattr( self, name )

   def __getattr2( self, name ):             # "running" getattr
    # handle "from ROOT import *" ... can be called multiple times
      if name == '__all__':
         if '__IPYTHON__' in __builtins__:
            import warnings
            warnings.warn( '"from ROOT import *" is not supported under IPython' )
            # continue anyway, just in case it works ...

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
      try:
         attr = _root.LookupCppEntity( name, PyConfig.ExposeCppMacros )
         if type(attr) == _root.PropertyProxy:
            setattr( self.__class__, name, attr )      # descriptor
            return getattr( self, name )
         else:
            self.__dict__[ name ] = attr               # normal member
            return attr
      except AttributeError:
         pass

    # reaching this point means failure ...
      raise AttributeError( name )

   def __delattr__( self, name ):
    # this is for convenience, as typically lookup results are kept at two places
      try:
         delattr( self.module._root, name )
      except AttributeError:
         pass

      return super( self.__class__, self ).__delattr__( name )

   def __finalSetup( self ):
    # prevent this method from being re-entered through the gROOT wrapper
      self.__dict__[ 'gROOT' ] = _root.gROOT

    # switch to running gettattr/setattr
      self.__class__.__getattr__ = self.__class__.__getattr2
      del self.__class__.__getattr2
      self.__class__.__setattr__ = self.__class__.__setattr2
      del self.__class__.__setattr2

    # normally, you'll want a ROOT application; don't init any further if
    # one pre-exists from some C++ code somewhere
      hasargv = hasattr( sys, 'argv' )
      if hasargv and PyConfig.IgnoreCommandLineOptions:
         argv = sys.argv
         sys.argv = []

      appc = _root.CreateScopeProxy( 'PyROOT::TPyROOTApplication' )
      if appc.CreatePyROOTApplication():
         appc.InitROOTGlobals()
         # TODO Cling equivalent needed: appc.InitCINTMessageCallback();
         appc.InitROOTMessageCallback();

      if hasargv and PyConfig.IgnoreCommandLineOptions:
         sys.argv = argv

    # must be called after gApplication creation:
      if '__IPYTHON__' in __builtins__:
       # IPython's FakeModule hack otherwise prevents usage of python from Cling (TODO: verify necessity)
         _root.gROOT.ProcessLine( 'TPython::Exec( "" );' )
         sys.modules[ '__main__' ].__builtins__ = __builtins__

    # special case for cout (backwards compatibility)
      if hasattr( cppyy.gbl.std, '__1' ):
         attr_1 = getattr( cppyy.gbl.std, '__1' )
         if hasattr( attr_1, 'cout' ):
            self.__dict__[ 'cout' ] = attr_1.cout

    # custom logon file (must be after creation of ROOT globals)
      if hasargv and not '-n' in sys.argv:
         rootlogon = os.path.expanduser( '~/.rootlogon.py' )
         if os.path.exists( rootlogon ):
          # could also have used execfile, but import is likely to give fewer surprises
            import imp
            imp.load_module( 'rootlogon', open( rootlogon, 'r' ), rootlogon, ('.py','r',1) )
            del imp
         else:  # if the .py version of rootlogon exists, the .C is ignored (the user can
                # load the .C from the .py, if so desired)

          # system logon, user logon, and local logon (skip Rint.Logon)
            name = '.rootlogon.C'
            logons = [ os.path.join( str(self.gRootDir), 'etc', 'system' + name ),
                       os.path.expanduser( os.path.join( '~', name ) ) ]
            if logons[-1] != os.path.join( os.getcwd(), name ):
               logons.append( name )
            for rootlogon in logons:
               if os.path.exists( rootlogon ):
                  appc.ExecuteFile( rootlogon )
            del rootlogon, logons

    # use either the input hook or thread to send events to GUIs
      if self.PyConfig.StartGuiThread and \
            not ( self.keeppolling or _root.gROOT.IsBatch() ):
         if self.PyConfig.StartGuiThread == 'inputhook' or\
               _root.gSystem.InheritsFrom( 'TMacOSXSystem' ):
          # new, PyOS_InputHook based mechanism
            if PyConfig.GUIThreadScheduleOnce:
               for guicall in PyConfig.GUIThreadScheduleOnce:
                  guicall()
               PyConfig.GUIThreadScheduleOnce = []
            _root.InstallGUIEventInputHook()
         else:
          # original, threading based approach
            import threading
            self.__dict__[ 'keeppolling' ] = 1
            self.__dict__[ 'PyGUIThread' ] = \
               threading.Thread( None, _processRootEvents, None, ( self, ) )

            def _finishSchedule( ROOT = self ):
               import threading
               if threading.currentThread() != self.PyGUIThread:
                  while self.PyConfig.GUIThreadScheduleOnce:
                     self.PyGUIThread.join( 0.1 )

            self.PyGUIThread.finishSchedule = _finishSchedule
            self.PyGUIThread.setDaemon( 1 )
            self.PyGUIThread.start()

    # store already available ROOT objects to prevent spurious lookups
      for name in self.module.__pseudo__all__ + _memPolicyAPI + _sigPolicyAPI:
         self.__dict__[ name ] = getattr( _root, name )

    # the macro NULL is not available from Cling globals, but might be useful
      setattr( _root, 'NULL', 0 )

      for name in cppyy.gbl.std.stlclasses:
         setattr( _root, name, getattr( cppyy.gbl.std, name ) )

    # set the display hook
      sys.displayhook = _displayhook


sys.modules[ __name__ ] = ModuleFacade( sys.modules[ __name__ ] )
del ModuleFacade


### b/c of circular references, the facade needs explicit cleanup ---------------
import atexit
def cleanup():
 # save for later
   isCocoa = _root.gSystem.InheritsFrom( 'TMacOSXSystem' )

 # restore hooks
   import sys
   sys.displayhook = sys.__displayhook__
   if not '__IPYTHON__' in __builtins__:
      sys.excepthook = sys.__excepthook__
   __builtin__.__import__ = _orig_ihook

   facade = sys.modules[ __name__ ]

 # shutdown GUI thread, as appropriate (always save to call)
   _root.RemoveGUIEventInputHook()

 # prevent further spurious lookups into ROOT libraries
   del facade.__class__.__getattr__
   del facade.__class__.__setattr__

 # shutdown GUI thread, as appropriate
   if hasattr( facade, 'PyGUIThread' ):
      facade.keeppolling = 0

    # if not shutdown from GUI (often the case), wait for it
      import threading
      if threading.currentThread() != facade.PyGUIThread:
         facade.PyGUIThread.join( 3. )                 # arbitrary
      del threading

 # remove otherwise (potentially) circular references
   import types
   items = facade.module.__dict__.items()
   for k, v in items:
      if type(v) == types.ModuleType:
         facade.module.__dict__[ k ] = None
   del v, k, items, types

 # destroy facade
   facade.__dict__.clear()
   del facade

 # run part the gROOT shutdown sequence ... running it here ensures that
 # it is done before any ROOT libraries are off-loaded, with unspecified
 # order of static object destruction; 
   gROOT = sys.modules[ 'libPyROOT' ].gROOT
   gROOT.EndOfProcessCleanups()
   del gROOT

 # cleanup cached python strings
   sys.modules[ 'libPyROOT' ]._DestroyPyStrings()

 # destroy ROOT extension module and ROOT module
   del sys.modules[ 'libPyROOT' ]
   del sys.modules[ 'ROOT' ]

atexit.register( cleanup )
del cleanup, atexit
