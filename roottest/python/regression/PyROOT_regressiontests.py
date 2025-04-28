# File: roottest/python/regression/PyROOT_regressiontests.py
# Author: Wim Lavrijsen (LBNL, WLavrijsen@lbl.gov)
# Created: 01/02/07
# Last: 04/26/16

"""Regression tests, lacking a better place, for PyROOT package."""

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
from ROOT import TH1I, TVector3, TGraph, TMatrixD
import cppyy

cleaned_preload = os.environ.get('LD_PRELOAD', None)

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
   'Regression18FailingDowncast',
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
      # See https://github.com/root-project/root/issues/7541 and
      # https://bugs.llvm.org/show_bug.cgi?id=49692 :
      # llvm JIT fails to catch exceptions on M1, so we disable their testing
      if platform.processor() != "arm" or platform.mac_ver()[0] == '':
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

   def test2TemplateWithNamespaceReturnValue(self):
      """Test the return value of a templated function in a namespace"""
      # ROOT-10256
      ROOT.gInterpreter.Declare("""
      namespace bar {
         template<typename T>
         T foo(T a) { return a; }
      }
      """)
      self.assertEqual(ROOT.bar.foo(1), 1)

   def test3TemplatedMethodWithReferenceParameter(self):
      """Test templated method with reference parameter"""
      # ROOT-10292
      ROOT.gInterpreter.Declare("""
      struct TriggerResults {};
      struct Tag {};

      template <typename T> struct HandleT {};

      struct Event {
         template <typename T> int getByLabel(const Tag&, HandleT<T>&) { return 0; }
      };
      """)

      ev = ROOT.Event()
      tag = ROOT.Tag()
      result = ROOT.HandleT(ROOT.TriggerResults)()

      self.assertEqual(ev.getByLabel(tag, result), 0)


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
      # Cppyy's Long is deprecated in favour of ctypes.c_long
      # https://bitbucket.org/wlav/cppyy/issues/101
      import ctypes
      self.assertEqual( ROOT.marsupilami, a.testEnum4( ctypes.c_long(ROOT.marsupilami).value ) )


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


### importing cout should not result in printed errors =======================
class Regression11GlobalsLookup( MyTestCase ):
   def test1GetCout( self ):
      """Test that ROOT.cout does not cause error messages"""

      import ROOT
      # Look for cout in std
      c = ROOT.std.cout

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


### const-ref passing differs between CINT and Cling =========================
class Regression15ConsRef( MyTestCase ):
   def test1PassByConstRef( self ):
      """Test passing arguments by const reference"""

      tnames = [ "bool", "short", "int", "long" ]
      for i in range(len(tnames)):
         gInterpreter.LoadText(
            "bool PyROOT_Regression_TakesRef%d(const %s& arg) { return arg; }" % (i, tnames[i]) )
         self.assertTrue( not eval( "ROOT.PyROOT_Regression_TakesRef%d(0)" % (i,) ) )
         self.assertTrue( eval( "ROOT.PyROOT_Regression_TakesRef%d(1)" % (i,) ) )
      self.assertEqual( len(tnames)-1, i )


### nested namespace had a bug in the lookup loop ============================
class Regression16NestedNamespace( MyTestCase ):
   def test1NestedNamespace( self ):
      """Test nested namespace lookup"""

      gROOT.ProcessLine('#include "NestedNamespace.h"')
      self.assertTrue( ROOT.ABCDEFG.ABCD.Nested )


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

### Reuse of Python proxies in attribute lookups ================
class Regression21ReuseProxies(MyTestCase):
   def test1ReuseProxies(self):
      """Test that Python proxies are reused in attribute lookups"""
      # ROOT-8843
      import ROOT
      ROOT.gInterpreter.LoadText("struct A { A* otherA=nullptr;};")
      a1 = ROOT.A()
      a2 = ROOT.A()
      a1.otherA = a2
      a3 = a1.otherA
      self.assertEqual(a3, a2)
      val = 4
      a3.b = val
      self.assertEqual(a2.b, val)
      self.assertEqual(a1.otherA.b, val)

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
   def test01DeletedCopyConstructor(self):
      """Test that deleted base class copy constructor is not used"""
      # ROOT-10872
      cppyy.gbl.gInterpreter.Declare('''
      struct NoCopy1 {
         NoCopy1() = default;
         NoCopy1(const NoCopy1&) = delete;
         virtual ~NoCopy1() = default;
      };

      struct MyClass1 : NoCopy1 {};
      ''')

      class MyDerived1(cppyy.gbl.MyClass1):
         pass

   def test02MoveConstructor(self):
      """Test that move constructor is not mistaken for copy constructor"""
      # ROOT-10872
      cppyy.gbl.gInterpreter.Declare('''
      struct NoCopy2 {
         NoCopy2() = default;
         NoCopy2(const NoCopy2&) = delete;
         NoCopy2(NoCopy2&&) = default;
         virtual ~NoCopy2() = default;
      };

      struct MyClass2 : NoCopy2 {};
      ''')

      class MyDerived2(cppyy.gbl.MyClass2):
         pass

   def test03ProtectedMethod(self):
       """Test that protected method is injected in derived class without crash"""
       # ROOT-10872
       ROOT.gInterpreter.Declare("""
       class CppAlg {
       public:
           virtual ~CppAlg() {}
       protected:
           int protectedMethod() { return 1; }
       };
       """)

       class Alg(ROOT.CppAlg): pass

       a = Alg()
       self.assertEqual(a.protectedMethod(), 1)

   def test04DerivedObjectDeletion(self):
       """Test that derived object is deleted without a crash"""
       # ROOT-11010
       ROOT.gInterpreter.Declare("""
       #include <string>

       class CppAlg2 {
       public:
           CppAlg2(std::string name) : m_name(name) {}
           virtual ~CppAlg2() {}
       private:
           std::string m_name;
       };
       """)

       class Alg2(ROOT.CppAlg2):
           def __init__(self, name):
               super(Alg2, self).__init__(name)

       a = Alg2('MyAlg')
       del a   # should not crash

   def test05BaseAndDerivedConstruction(self):
       """Test that creation of base class object does not interfere with creation of derived"""
       # ROOT-10789
       ROOT.gInterpreter.Declare("""
       #include <string>

       class CppAlg3 {
       public:
           CppAlg3(std::string name) : m_name(name) {}
           virtual ~CppAlg3() {}
           std::string m_name;
       };
       """)

       b = ROOT.CppAlg3("MyAlgBase")

       class Alg3(ROOT.CppAlg3):
           def __init__(self, name):
               super(Alg3, self).__init__(name)

       test = 'MyAlgDerived'
       d = Alg3(test)
       self.assertEqual(test, d.m_name)

       class Alg3_2(ROOT.CppAlg3):
           pass

       d2 = Alg3_2(test)
       self.assertEqual(test, d2.m_name)

   def test06MultiInheritance(self):
       """Test for a Python derived class in presence of multiple inheritance in C++"""
       # 6376
       cppyy.gbl.gInterpreter.Declare("""
       #include <array>
       #include <iostream>

       struct Interface1 {
         virtual int do_1()   = 0;
         virtual ~Interface1() = default;
       };

       struct Interface2 {
         virtual int do_2()   = 0;
         virtual ~Interface2() = default;
       };

       struct Base : virtual public Interface1, virtual public Interface2 {};

       struct Derived : Base, virtual public Interface2 {
         int do_1() override { return 1; }
         int do_2() override { return 2; }
       };

       int my_func( Interface2* i ) { return i->do_2(); }
       """)

       class PyDerived(cppyy.gbl.Derived): pass

       i = PyDerived()
       self.assertEqual(i.do_1(), 1)
       self.assertEqual(i.do_2(), 2)

       # Check there is no corruption in the invocation of i->do_2() inside my_func
       self.assertEqual(cppyy.gbl.my_func(i), 2)

   def test07ConstructorDefaultArgs(self):
       """Invocation of constructor with default arguments"""
       # 6467
       class MyTChain(ROOT.TChain):
          def __init__(self, name):
              # Invoke TChain(const char *name, const char *title="") constructor
              super(MyTChain, self).__init__(name)

       a = MyTChain("myname")

       # Try also without redefining __init__
       class MyTChain2(ROOT.TChain): pass

       b = MyTChain2("myname")

       self.assertEqual(a.GetName(), b.GetName())

   def test08ConstructorAllDefaultPars(self):
       """Invocation of a constructor that has default values for all its parameters"""
       # 6578
       class pMainFrame(ROOT.TGMainFrame):
           def __init__(self, parent, width, height ):
               ROOT.TGMainFrame.__init__(self, parent, width, height)

       window = pMainFrame(ROOT.gClient.GetRoot(), 200, 200)

   def test09MultipleProtectedAndPrivateOverloads(self):
       """Presence of multiple protected overloads of a method and both private and protected"""
       # 6345

       cppyy.gbl.gInterpreter.Declare('''
       class MyClass6345 {
       public:
          virtual ~MyClass6345() {}
       protected:
          int foo(int)      { return 1; }
          int foo(int, int) { return 2; }

          int bar()    { return 3; }
       private:
          int bar(int) { return 4; }
       };
       ''')

       class MyPyClass6345(cppyy.gbl.MyClass6345):
          pass

       a = MyPyClass6345()
       self.assertEqual(a.foo(0), 1)
       self.assertEqual(a.foo(0,0), 2)
       self.assertEqual(a.bar(), 3)
       self.assertRaises(TypeError, a.bar, 0)

   def test10DeepHierarchyVirtualCall(self):
       """Virtual call resolution in deep hierarchy (C++->Py->Py)"""
       # 6470

       cppyy.gbl.gInterpreter.Declare('''
       class CppBase6470 {
       public:
          virtual ~CppBase6470() {}
          virtual int fun1() const { return 1; }
          virtual int fun2() const { return fun1(); }
       };
       ''')

       CppBase = cppyy.gbl.CppBase6470

       class PyA(CppBase):
          def fun1(self): return 11

       class PyB(PyA):
          def fun1(self): return 111

       class PyC(PyA):
          pass

       base = CppBase()
       a = PyA()
       b = PyB()
       c = PyC()

       self.assertEqual(base.fun2(), 1)
       self.assertEqual(a.fun2(), 11)
       self.assertEqual(b.fun2(), 111)
       self.assertEqual(c.fun2(), 11) # PyA's fun1 should be invoked

   def test11MultiInheritancePyCpp(self):
       """Multiple inheritance from Python and Cpp classes"""

       class PurePy1:
          def foo(self): return 1

       class PurePy2:
          def bar(self): return 2

       cppyy.gbl.gInterpreter.Declare('''
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
