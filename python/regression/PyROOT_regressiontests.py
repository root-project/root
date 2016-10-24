# File: roottest/python/regression/PyROOT_regressiontests.py
# Author: Wim Lavrijsen (LBNL, WLavrijsen@lbl.gov)
# Created: 01/02/07
# Last: 04/26/16

"""Regression tests, lacking a better place, for PyROOT package."""

import sys, os, unittest
sys.path.append( os.path.join( os.getcwd(), os.pardir ) )

try:
   import commands
   WEXITSTATUS = os.WEXITSTATUS
except ImportError:
   import subprocess as commands
   def WEXITSTATUS(arg): return arg
import ROOT
from ROOT import gROOT, TClass, TObject, TH1I, TVector3, TGraph, PyROOT, Long, TFile, TMatrixD
from common import *

__all__ = [
   'Regression01TwiceImportStar',
   'Regression02PyException',
   'Regression03OldCrashers',
   'Regression04Threading',
   'Regression05LoKiNamespace',
   'Regression06Int64Conversion',
   'Regression07MatchConstWithProperReturn',
   'Regression08CheckEnumExactMatch',
   'Regression09TVector3Pythonize',
   'Regression10CoralAttributeListIterators',
   'Regression11GlobalsLookup',
   'Regression12WriteTGraph',
   'Regression13BaseClassUsing',
   'Regression14TPyException',
   'Regression15ConsRef',
   'Regression16NestedNamespace',
   'Regression17MatrixD',
   'Regression18FailingDowncast'
]


### "from ROOT import *" done in import-*-ed module ==========================
from Amir import *

class Regression01TwiceImportStar( MyTestCase ):
   def test1FromROOTImportStarInModule( self ):
      """Test handling of twice 'from ROOT import*'"""

      x = TestTChain()        # TestTChain defined in Amir.py


### TPyException thrown from C++ code ========================================
class Regression02PyException( MyTestCase ):
   def test1RaiseAndTrapPyException( self ):
      """Test thrown TPyException object processing"""

      # re-enabled as there are still issues with exceptions on Mac, and linker
      # issues on some of the test machines
      if FIXCLING:
         return

      gROOT.LoadMacro( "Scott.C+" )

    # test of not overloaded global function
      self.assertRaises( SyntaxError, ThrowPyException )
      try:
         ThrowPyException()
      except SyntaxError:
         self.assertEqual( str(sys.exc_info()[1]), "test error message" )

    # test of overloaded function
      self.assertRaises( SyntaxError, MyThrowingClass.ThrowPyException, 1 )
      try:
         MyThrowingClass.ThrowPyException( 1 )
      except SyntaxError:
         self.assertEqual( str(sys.exc_info()[1]), "overloaded int test error message" )


### Several tests that used to cause crashes =================================
class Regression03OldCrashers( MyTestCase ):
   def test1CreateTemporary( self ):
      """Handling of a temporary for user defined operator new"""

      gROOT.LoadMacro( "MuonTileID.C+" )
      getID = ROOT.getID

      getID()
      getID()                 # used to crash

   def test2UsageOfTQClassInstance( self ):
      """Calls on a TQClass instance"""

      self.assertEqual( TClass.GetClass("TQClass").GetName(), "TQClass" )

   def test3UseNamespaceInIteratorPythonization( self ):
      """Classes with iterators in a namespace"""

      gROOT.LoadMacro( "Marco.C" )
      ns = ROOT.ns
      
      self.assert_( ns.MyClass )

   def test4VerifyNoLoop( self ):
      """Smart class that returns itself on dereference should not loop"""

      gROOT.LoadMacro( "Scott3.C+" )
      MyTooSmartClass = ROOT.MyTooSmartClass
      
      a = MyTooSmartClass()
      self.assertRaises( AttributeError, getattr, a, 'DoesNotExist' )

   def test5DirectMetaClassAccess( self ):
      """Direct access on the meta class"""

      self.assertRaises( AttributeError, getattr, TObject.__class__, "nosuch" )

   def test6InspectionOfTH1I( self ):
      """Inspect TH1I"""

    # access to data member fArray used to fail w/o error set; ROOT-7336
      import inspect
      inspect.getmembers(TH1I)


### Test the condition under which to (not) start the GUI thread =============
class Regression04Threading( MyTestCase ):

   hasThread = gROOT.IsBatch() and 5 or 6   # can't test if no display ...
   noThread  = 5
   
   def test1SpecialCasegROOT( self ):
      """Test the special role that gROOT plays vis-a-vis threading"""

      cmd = sys.executable + "  -c 'import sys, ROOT; ROOT.gROOT; %s "\
            "sys.exit( 5 + int(\"thread\" in ROOT.__dict__) )'"
      if self.hasThread == self.noThread:
         cmd += " - -b"

      stat, out = commands.getstatusoutput( cmd % "" )
      self.assertEqual( WEXITSTATUS(stat), self.noThread )

      stat, out = commands.getstatusoutput( cmd % "ROOT.gROOT.SetBatch( 1 );" )
      self.assertEqual( WEXITSTATUS(stat), self.noThread )

      stat, out = commands.getstatusoutput( cmd % "ROOT.gROOT.SetBatch( 0 );" )
      self.assertEqual( WEXITSTATUS(stat), self.noThread )

      stat, out = commands.getstatusoutput(
         cmd % "ROOT.gROOT.ProcessLine( \"cout << 42 << endl;\" ); " )
      self.assertEqual( WEXITSTATUS(stat), self.hasThread )

      stat, out = commands.getstatusoutput( cmd % "ROOT.gDebug;" )
      self.assertEqual( WEXITSTATUS(stat), self.hasThread )

   def test2ImportStyles( self ):
      """Test different import styles vis-a-vis threading"""

      cmd = sys.executable + " -c 'import sys; %s ;"\
            "import ROOT; sys.exit( 5 + int(\"thread\" in ROOT.__dict__) )'"
      if self.hasThread == self.noThread:
         cmd += " - -b"

      stat, out = commands.getstatusoutput( cmd % "from ROOT import *" )
      self.assertEqual( WEXITSTATUS(stat), self.hasThread )

      stat, out = commands.getstatusoutput( cmd % "from ROOT import gROOT" )
      self.assertEqual( WEXITSTATUS(stat), self.noThread )

      stat, out = commands.getstatusoutput( cmd % "from ROOT import gDebug" )
      self.assertEqual( WEXITSTATUS(stat), self.hasThread )

   def test3SettingOfBatchMode( self ):
      """Test various ways of preventing GUI thread startup"""

      cmd = sys.executable + " -c '%s import ROOT, sys; sys.exit( 5+int(\"thread\" in ROOT.__dict__ ) )'"
      if self.hasThread == self.noThread:
         cmd += " - -b"

      stat, out = commands.getstatusoutput( (cmd % 'from ROOT import *;') + ' - -b' )
      self.assertEqual( WEXITSTATUS(stat), self.noThread )

      stat, out = commands.getstatusoutput(
         cmd % 'import ROOT; ROOT.PyConfig.StartGuiThread = 0;' )
      self.assertEqual( WEXITSTATUS(stat), self.noThread )

      stat, out = commands.getstatusoutput(
         cmd % 'from ROOT import PyConfig; PyConfig.StartGuiThread = 0; from ROOT import gDebug;' )
      self.assertEqual( WEXITSTATUS(stat), self.noThread )

      stat, out = commands.getstatusoutput(
         cmd % 'from ROOT import PyConfig; PyConfig.StartGuiThread = 1; from ROOT import gDebug;' )
      self.assertEqual( WEXITSTATUS(stat), self.hasThread )

      stat, out = commands.getstatusoutput(
         cmd % 'from ROOT import gROOT; gROOT.SetBatch( 1 ); from ROOT import *;' )
      self.assertEqual( WEXITSTATUS(stat), self.noThread )

      if not gROOT.IsBatch():               # can't test if no display ...
         stat, out = commands.getstatusoutput(
            cmd % 'from ROOT import gROOT; gROOT.SetBatch( 0 ); from ROOT import *;' )
         self.assertEqual( WEXITSTATUS(stat), self.hasThread )


### Test the proper resolution of a template with namespaced parameter =======
class Regression05LoKiNamespace( MyTestCase ):
   def test1TemplateWithNamespaceParameter( self ):
      """Test name resolution of template with namespace parameter"""

      rcp = 'const LHCb::Particle*'

      gROOT.LoadMacro( 'LoKiNamespace.C+' )
      LoKi = ROOT.LoKi

      self.assertEqual( LoKi.Constant( rcp ).__name__, 'Constant<%s>' % rcp )
      self.assertEqual(
         LoKi.BooleanConstant( rcp ).__name__, 'BooleanConstant<%s>' % rcp )

### Test conversion of int64 objects to ULong64_t and ULong_t ================
class Regression06Int64Conversion( MyTestCase ):
   limit1  = 4294967295
   limit1L = pylong(4294967295)

   def test1IntToULongTestCase( self ):
      """Test conversion of Int(64) limit values to unsigned long"""

      gROOT.LoadMacro( 'ULongLong.C+' )
      ULongFunc = ROOT.ULongFunc

      self.assertEqual( self.limit1,  ULongFunc( self.limit1 ) )
      self.assertEqual( self.limit1L, ULongFunc( self.limit1 ) )
      self.assertEqual( self.limit1L, ULongFunc( self.limit1L ) )
      self.assertEqual( maxvalue + 2, ULongFunc( maxvalue + 2 ) )

   def test2IntToULongLongTestCase( self ):
      """Test conversion of Int(64) limit values to unsigned long long"""
      ULong64Func = ROOT.ULong64Func

      self.assertEqual( self.limit1,  ULong64Func( self.limit1 ) )
      self.assertEqual( self.limit1L, ULong64Func( self.limit1 ) )
      self.assertEqual( self.limit1L, ULong64Func( self.limit1L ) )
      self.assertEqual( maxvalue + 2, ULong64Func( maxvalue + 2 ) )


### Proper match-up of return type and overloaded function ===================
class Regression07MatchConstWithProperReturn( MyTestCase ):
   def test1OverloadOrderWithProperReturn( self ):
      """Test return type against proper overload w/ const and covariance"""

      gROOT.LoadMacro( "Scott2.C+" )
      MyOverloadOneWay = ROOT.MyOverloadOneWay
      MyOverloadTheOtherWay = ROOT.MyOverloadTheOtherWay

      self.assertEqual( MyOverloadOneWay().gime(), 1 )
      self.assertEqual( MyOverloadTheOtherWay().gime(), "aap" )


### enum type conversions (used to fail exact match in CINT) =================
class Regression08CheckEnumExactMatch( MyTestCase ):
   def test1CheckEnumCalls( self ):
      """Be able to pass enums as function arguments"""

      gROOT.LoadMacro( "Till.C+" )
     
      a = ROOT.Monkey()
      self.assertEqual( ROOT.fish, a.testEnum1( ROOT.fish ) )
      self.assertEqual( ROOT.cow,  a.testEnum2( ROOT.cow ) )
      self.assertEqual( ROOT.bird, a.testEnum3( ROOT.bird ) )
      self.assertEqual( ROOT.marsupilami, a.testEnum4( ROOT.marsupilami ) )
      self.assertEqual( ROOT.marsupilami, a.testEnum4( Long(ROOT.marsupilami) ) )


### test pythonization of TVector3 ===========================================
class Regression09TVector3Pythonize( MyTestCase ):
   def test1TVector3( self ):
      """Verify TVector3 pythonization"""

      v = TVector3( 1., 2., 3.)
      self.assertEqual( list(v), [1., 2., 3. ] )

      w = 2*v
      self.assertEqual( w.x(), 2*v.x() )
      self.assertEqual( w.y(), 2*v.y() )
      self.assertEqual( w.z(), 2*v.z() )


### test pythonization coral::AttributeList iterators ========================
class Regression10CoralAttributeListIterators( MyTestCase ):
   def test1IterateWithBaseIterator( self ):
      """Verify that the correct base class iterators is picked up"""

      gROOT.LoadMacro( "CoralAttributeList.C+" )
      coral_pyroot_regression = ROOT.coral_pyroot_regression

      a = coral_pyroot_regression.AttributeList()

      a.extend( "i", "int" )
      self.assertEqual( a.size(), 1 )
      self.assertEqual( a.begin(), a.begin() )
      self.assertNotEqual( a.begin(), a.end() )

      b = a.begin()
      e = a.end()
      self.assertNotEqual( a, e )

      b.__preinc__()
      self.assertEqual( b, e )
      self.assertNotEqual( b, a.begin() )


### importing cout should not result in printed errors =======================
class Regression11GlobalsLookup( MyTestCase ):
   def test1GetCout( self ):
      """Test that ROOT.cout does not cause error messages"""

      import ROOT
      c = ROOT.cout

   def test2GlobalFromROOTNamespace( self ):
      """Entities in 'ROOT::' need no explicit 'ROOT.'"""

      import ROOT
      m = ROOT.Math


### importing cout should not result in printed errors =======================
class Regression12WriteTGraph( MyTestCase ):
   def test1WriteTGraph( self ):
      """Write a TGraph object and read it back correctly"""

      gr = TGraph()
      ff = TFile( "test.root", "RECREATE" )
      ff.WriteObject( gr, "grname", "" )
      gr2 = TGraph()
      ff.GetObject( "grname", gr2 )
      os.remove( "test.root" )


### 'using' base class data members should make them accessible ==============
class Regression13BaseClassUsing( MyTestCase ):
   def test1AccessUsingBaseClassDataMember( self ):
      """Access a base class data member made availabe by 'using'"""

      if FIXCLING:
         return

      p = TPySelector()
      str( p.fInput )        # segfaults in case of failure


### TPyException had troubles due to its base class of std::exception ========
class Regression14TPyException( MyTestCase ):
   def test1PythonAccessToTPyException( self ):
      """Load TPyException into python and make sure its usable"""

      e = PyROOT.TPyException()
      self.assert_( e )
      self.assertEqual( e.what(), "python exception" )


### const-ref passing differs between CINT and Cling =========================
class Regression15ConsRef( MyTestCase ):
   def test1PassByConstRef( self ):
      """Test passing arguments by const reference"""

      tnames = [ "bool", "short", "int", "long" ]
      for i in range(len(tnames)):
         gInterpreter.LoadText(
            "bool PyROOT_Regression_TakesRef%d(const %s& arg) { return arg; }" % (i, tnames[i]) )
         self.assert_( not eval( "ROOT.PyROOT_Regression_TakesRef%d(0)" % (i,) ) )
         self.assert_( eval( "ROOT.PyROOT_Regression_TakesRef%d(1)" % (i,) ) )
      self.assertEqual( len(tnames)-1, i )


### nested namespace had a bug in the lookup loop ============================
class Regression16NestedNamespace( MyTestCase ):
   def test1NestedNamespace( self ):
      """Test nested namespace lookup"""

      gROOT.ProcessLine('#include "NestedNamespace.h"')
      self.assert_( ROOT.ABCDEFG.ABCD.Nested )


### matrix access has to go through non-const lookup =========================
class Regression17MatrixD( MyTestCase ):
   def test1MatrixElementAssignment( self ):
      """Matrix lookup has to be non-const to allow assigment"""

      m = TMatrixD( 5, 5 )
      self.assert_( not 'const' in type(m[0]).__name__ )

    # test assignment
      m[1][2] = 3.
      self.assertEqual( m[1][2], 3. )

      m[1, 2] = 4.
      self.assertEqual( m[1][2], 4. )


### classes weren't always classes making GetActualClass fail ================
class Regression18FailingDowncast( MyTestCase ):
   def test1DowncastOfInterpretedClass( self ):
      """Auto-downcast of interpreted class"""

      code = """namespace RG18 {
class Base {
public:
  virtual ~Base(){}
};

class Derived : public Base {
  virtual ~Derived() {}
};

Base* g() { return new Derived(); }
}"""
      gInterpreter.LoadText( code )

      self.assertEqual( type(ROOT.RG18.g()), ROOT.RG18.Derived )


## actual test run
if __name__ == '__main__':
   from MyTextTestRunner import MyTextTestRunner

   loader = unittest.TestLoader()
   testSuite = loader.loadTestsFromModule( sys.modules[ __name__ ] )

   runner = MyTextTestRunner( verbosity = 2 )
   result = not runner.run( testSuite ).wasSuccessful()

   sys.exit( result )
