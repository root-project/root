# File: roottest/python/regression/PyROOT_regressiontests.py
# Author: Wim Lavrijsen (LBNL, WLavrijsen@lbl.gov)
# Created: 01/02/07
# Last: 04/26/16

"""Regression tests, lacking a better place, for PyROOT package.

NOTE: several of the original test cases in this file were removed because
they are covered upstream in the cppyy test suite
(bindings/pyroot/cppyy/cppyy/test/): see test_advancedcpp.py,
test_templates.py, test_datatypes.py, test_fragile.py, test_streams.py,
test_regression.py and test_crossinheritance.py. What remains are
ROOT-specific tests and tests without upstream equivalents.
"""

import platform
import sys, os, unittest
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

if not os.path.exists('ScottCppyy.C'):
    os.chdir(os.path.dirname(__file__))

try:
   import commands
   WEXITSTATUS = os.WEXITSTATUS
except ImportError:
   import subprocess as commands
   def WEXITSTATUS(arg): return arg

original_preload = os.environ.get('LD_PRELOAD', None)

import ROOT
ROOT.PyConfig.IgnoreCommandLineOptions = False
from ROOT import gROOT, gInterpreter
from ROOT import TClass, TObject, TFile
from ROOT import TVector3, TGraph, TMatrixD
import cppyy

cleaned_preload = os.environ.get('LD_PRELOAD', None)

from common import *

__all__ = [
   'Regression01TwiceImportStar',
   'Regression02PyException',
   'Regression03OldCrashers',
   'Regression04Threading',
   'Regression09TVector3Pythonize',
   'Regression10CoralAttributeListIterators',
   'Regression11GlobalsLookup',
   'Regression12WriteTGraph',
   'Regression14TPyException',
   'Regression17MatrixD',
]


### "from ROOT import *" done in import-*-ed module ==========================
from Amir import *


class Regression01TwiceImportStar( MyTestCase ):
   def test1FromROOTImportStarInModule( self ):
      """Test handling of twice 'from ROOT import*'"""

      x = TestTChain()        # TestTChain defined in Amir.py


# TPyException thrown from C++ code ========================================
class Regression02PyException(MyTestCase):
   def test1RaiseAndTrapPyException(self):
      """Test thrown TPyException object processing"""
      gROOT.LoadMacro("ScottCppyy.C+")

      # test of not overloaded global function
      with self.assertRaisesRegex(SyntaxError, "test error message"):
         ROOT.ThrowPyException()

      # test of overloaded function
      with self.assertRaisesRegex(SyntaxError, "overloaded int test error message"):
         ROOT.MyThrowingClass.ThrowPyException(1)


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

      self.assertTrue( ns.MyClass )

   def test4VerifyNoLoop( self ):
      """Smart class that returns itself on dereference should not loop"""

      gROOT.LoadMacro( "Scott3.C+" )
      MyTooSmartClass = ROOT.MyTooSmartClass

      a = MyTooSmartClass()
      self.assertRaises( AttributeError, getattr, a, 'DoesNotExist' )

   def test5DirectMetaClassAccess( self ):
      """Direct access on the meta class"""

      self.assertRaises( AttributeError, getattr, TObject.__class__, "nosuch" )


### Test the condition under which to (not) start the GUI thread =============
class Regression04Threading( MyTestCase ):

   hasThread = gROOT.IsBatch() and 5 or 6   # can't test if no display ...
   noThread  = 5

   def test1SpecialCasegROOT( self ):
      """Test the special role that gROOT plays vis-a-vis threading"""

      # Restore original LD_PRELOAD when running under AddressSanitizer
      if original_preload is not None:
         os.environ['LD_PRELOAD'] = original_preload

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

      # Restore the cleaned LD_PRELOAD for other tests
      if cleaned_preload is not None:
         os.environ['LD_PRELOAD'] = cleaned_preload

   def test2ImportStyles( self ):
      """Test different import styles vis-a-vis threading"""

      # Restore original LD_PRELOAD when running under AddressSanitizer
      if original_preload is not None:
         os.environ['LD_PRELOAD'] = original_preload

      cmd = sys.executable + " -c 'import sys; %s ;"\
            "import ROOT; sys.exit( 5 + int(\"thread\" in ROOT.__dict__) )'"
      if self.hasThread == self.noThread:
         cmd += " - -b"

      stat, out = commands.getstatusoutput( cmd % "from ROOT import gROOT" )
      self.assertEqual( WEXITSTATUS(stat), self.noThread )

      stat, out = commands.getstatusoutput( cmd % "from ROOT import gDebug" )
      self.assertEqual( WEXITSTATUS(stat), self.hasThread )

      # Restore the cleaned LD_PRELOAD for other tests
      if cleaned_preload is not None:
         os.environ['LD_PRELOAD'] = cleaned_preload

   def test3SettingOfBatchMode( self ):
      """Test various ways of preventing GUI thread startup"""

      # Restore original LD_PRELOAD when running under AddressSanitizer
      if original_preload is not None:
         os.environ['LD_PRELOAD'] = original_preload

      cmd = sys.executable + " -c '%s import ROOT, sys; sys.exit( 5+int(\"thread\" in ROOT.__dict__ ) )'"
      if self.hasThread == self.noThread:
         cmd += " - -b"

      stat, out = commands.getstatusoutput(
         cmd % 'import ROOT; ROOT.PyConfig.StartGuiThread = 0;' )
      self.assertEqual( WEXITSTATUS(stat), self.noThread )

      stat, out = commands.getstatusoutput(
         cmd % 'from ROOT import PyConfig; PyConfig.StartGuiThread = 0; from ROOT import gDebug;' )
      self.assertEqual( WEXITSTATUS(stat), self.noThread )

      stat, out = commands.getstatusoutput(
         cmd % 'from ROOT import PyConfig; PyConfig.StartGuiThread = 1; from ROOT import gDebug;' )
      self.assertEqual( WEXITSTATUS(stat), self.hasThread )

      # Restore the cleaned LD_PRELOAD for other tests
      if cleaned_preload is not None:
         os.environ['LD_PRELOAD'] = cleaned_preload


### test pythonization and operators of TVector3 ===========================================
class Regression09TVector3Pythonize( MyTestCase ):
   def test1TVector3( self ):
      """Verify TVector3 pythonization"""

      v = TVector3( 1., 2., 3.)
      self.assertEqual( list(v), [1., 2., 3. ] )

      w = 2*v
      self.assertEqual( w.x(), 2*v.x() )
      self.assertEqual( w.y(), 2*v.y() )
      self.assertEqual( w.z(), 2*v.z() )

   def test2TVector3(self):
      """Verify that using one operator* overload does not mask the others"""
      # ROOT-10278
      v = TVector3(1., 2., 3.)
      v*2
      self.assertEqual(v*v, 14.0)


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


### entities in the ROOT:: namespace =========================================
class Regression11GlobalsLookup( MyTestCase ):
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
      # In new PyROOT, use a nicer way to get objects in files,
      # the TDirectory::Get() pythonisation:
      ff.Get("grname")
      os.remove( "test.root" )


### TPyException had troubles due to its base class of std::exception ========
class Regression14TPyException( MyTestCase ):
   def test1PythonAccessToTPyException( self ):
      """Load TPyException into python and make sure its usable"""

      # In exp PyROOT, TPyException is called PyException and it belongs
      # to the CPyCppyy namespace.
      # Also, it is not included in the PCH, so we need to include the
      # header first
      ROOT.gInterpreter.Declare("#include \"CPyCppyy/PyException.h\"")
      e = ROOT.CPyCppyy.PyException()
      self.assertTrue( e )
      self.assertEqual( e.what(), "python exception" )


### matrix access has to go through non-const lookup =========================
class Regression17MatrixD( MyTestCase ):
   def test1MatrixElementAssignment( self ):
      """Matrix lookup has to be non-const to allow assigment"""

      m = TMatrixD( 5, 5 )
      self.assertTrue( not 'const' in type(m[0]).__name__ )

    # test assignment
      m[1][2] = 3.
      self.assertEqual( m[1][2], 3. )

      m[1, 2] = 4.
      self.assertEqual( m[1][2], 4. )


### Tests for TGL classes ================
try:
   from ROOT import TGLLine3, TGLVertex3, TGLVector3
except ImportError:
   print("GL classes not found, skipping GL tests")
else:
   class Regression19TGL(MyTestCase):
      def test1TGLVertex3OperatorPlus(self):
         """Try invoking TGLVertex3::operator+ twice"""
         # ROOT-10166
         scatteringPoint = TGLVertex3(2., 3., 0.2)
         glvec3 = TGLVector3(1,2,3)

         vertexEnd = scatteringPoint + glvec3
         vertexEnd = scatteringPoint + glvec3

      def test2TGLLine3Constructor(self):
         """Check that the right constructor of TGLLine3 is called"""
         # ROOT-10102
         trackAfterScattering = TGLLine3(TGLVertex3(2., 3., 0.2), TGLVector3(0., 0., -20.))

         self.assertEqual(trackAfterScattering.Vector().X(), .0)
         self.assertEqual(trackAfterScattering.Vector().Y(), .0)
         self.assertEqual(trackAfterScattering.Vector().Z(), -20.0)


### Getting and setting configuration options of gEnv ================
class Regression20gEnv(MyTestCase):
   def test1GetSetValue(self):
      """Set a value with gEnv and retrieve it afterwards"""
      # ROOT-10155
      from ROOT import gEnv

      optname = "SomeOption"
      defval = -1
      self.assertEqual(gEnv.GetValue(optname, defval), defval)
      newval = 0
      gEnv.SetValue(optname, newval)
      self.assertEqual(gEnv.GetValue(optname, defval), newval)

### Tests related to cleanup of proxied objects ================
class Regression22ObjectCleanup(MyTestCase):
   def test1GetListOfGraphs(self):
      """List returned by GetListOfGraphs should not have kMustCleanup set to true"""
      # ROOT-9040
      mg = ROOT.TMultiGraph()
      tg = ROOT.TGraph()
      # The TMultiGraph will take the ownership of the added TGraphs
      ROOT.SetOwnership(tg, False)
      mg.Add(tg)

      l = mg.GetListOfGraphs()
      self.assertEqual(l.TestBit(ROOT.kMustCleanup), False)

      c = ROOT.TCanvas()
      mg.Draw()


class Regression23TFractionFitter(MyTestCase):
   def test1TFractionFitterDestruction(self):
      """Test proper destruction of TFractionFitter object"""
      # ROOT-9414
      h1 = ROOT.TH1F("h1","h1",1,0,1)
      h2 = ROOT.TH1F("h2","h2",1,0,1)
      h3 = ROOT.TH1F("h3","h3",1,0,1)

      h1.Fill(0.5)
      h2.Fill(0.5)
      h3.Fill(0.5)
      h3.Fill(0.5)

      mc = ROOT.TObjArray(2)
      mc.Add(h1)
      mc.Add(h2)

      ff = ROOT.TFractionFitter(h3, mc)
      ff.Fit()


class Regression24CppPythonInheritance(MyTestCase):
   # NOTE: the former tests 01-07, 09 and 10 of this class were removed: they
   # are covered upstream in the cppyy test suite by test_crossinheritance.py
   # (test24_non_copyable, test14_protected_access, test13_virtual_dtors_and_del,
   # test35_deletion, test02_constructor, test19/test20 multiple inheritance,
   # test22_multiple_inheritance_with_defaults, test30_access_and_overload,
   # test17_deep_hierarchy, test28_cross_deep).

   def test08ConstructorAllDefaultPars(self):
       """Invocation of a constructor that has default values for all its parameters"""
       # 6578
       class pMainFrame(ROOT.TGMainFrame):
           def __init__(self, parent, width, height ):
               ROOT.TGMainFrame.__init__(self, parent, width, height)

       window = pMainFrame(ROOT.gClient.GetRoot(), 200, 200)

   def test11MultiInheritancePyCpp(self):
       """Multiple inheritance from Python and Cpp classes"""

       class PurePy1:
          def foo(self): return 1

       class PurePy2:
          def bar(self): return 2

       cppyy.cppdef('''
       class MyCppClass11 {
       public:
          int foo() { return 3; }
          int bar() { return 4; }
          virtual ~MyCppClass11() {}
       };
       ''')

       # Multiple Python classes and just one C++ class are supported
       class PyDerived11(PurePy1, cppyy.gbl.MyCppClass11, PurePy2): pass

       x = PyDerived11()
       self.assertEqual(x.foo(), 1) # PurePy1's foo
       self.assertEqual(x.bar(), 4) # MyCppClass11's bar


class Regression25MapGetItemToCall(MyTestCase):
   def test01MapGetItemToCall(self):
       """Test reduction of range of mapping __getitem__ to __call__"""
       # 7179

       cppyy.cppdef('''
       struct Foo7179 {
          float operator[](float x) {
             return x;
          }
       };

       struct Bar7179 : public Foo7179 {
          float operator()(float x) {
             return -1;
          }
       };
       ''')

       b = cppyy.gbl.Bar7179()
       self.assertEqual(b[42], 42)

class Regression26OverloadedOperator( MyTestCase ):
   def test1CheckOverloadedOperator( self ):
      """Test accessibility of derived class in presence of an overloaded operator()"""

      code = """
      namespace Regression26 {
         struct DCBase
         {
            double& operator()();
         };
         struct DenseBase : public DCBase
         {
         using DCBase::operator();
         template <typename T> int operator()() const;
         };
      }
      """
      gInterpreter.LoadText(code)
      foo = ROOT.Regression26.DenseBase


class Regression27ImplicitSmartPtrOverload(MyTestCase):
    def test1CheckImplicitSmartPtrOverload(self):
        """Implicit smart pointer conversion should not cause wrong overload choice.

        Test that if there are both smart pointer and rvalue reference overloads,
        the smart pointer overload is not chosen by accident.

        Covers GitHub issue https://github.com/root-project/root/issues/15117."""

        ROOT.gInterpreter.LoadText(
            """
        namespace regression27 {

        struct Base {
           virtual ~Base() = default;
           virtual int func() const = 0;
        };

        struct Derived : public Base {
           Derived(int i) : m_i(i) {}
           ~Derived() = default;
           Derived(const Derived &) = delete;
           Derived &operator=(const Derived &) = delete;
           Derived(Derived &&) = default;
           Derived &operator=(Derived &&) = default;

           int func() const final { return m_i; }

        private:
           int m_i{42};
        };

        int foo(std::unique_ptr<Base> basePtr)
        {
           return 1;
        }

        template <typename T,
                  typename = std::enable_if_t<std::is_base_of_v<Base, T> && !std::is_lvalue_reference_v<T>>>
        int foo(T &&t)
        {
           return 2;
        }

        } // namespace regression27
        """
        )

        c = ROOT.regression27.Derived(123)
        self.assertEqual(ROOT.regression27.foo(ROOT.std.move(c)), 2)  # we expect the second overload


## actual test run
if __name__ == '__main__':
   from MyTextTestRunner import MyTextTestRunner

   loader = unittest.TestLoader()
   testSuite = loader.loadTestsFromModule( sys.modules[ __name__ ] )

   runner = MyTextTestRunner( verbosity = 2 )
   result = not runner.run( testSuite ).wasSuccessful()

   sys.exit( result )
