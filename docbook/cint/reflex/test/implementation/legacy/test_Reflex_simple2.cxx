// @(#)root/reflex:$Id$
// Author: Stefan Roiser 2004

// CppUnit include file
#include "cppunit/extensions/HelperMacros.h"

// Seal include files
#include "Reflex/Reflex.h"

// Standard C++ include files
#include <iostream>
#ifdef _WIN32
  # include <windows.h>
# ifdef CONST
#  undef CONST
# endif
#else
  # include <dlfcn.h>
#endif

using namespace std;
using namespace ROOT::Reflex;
enum Visibility { Public, Protected, Private };

void
generate_class_decl(const Type& cl,
                    const string& indent) {
   // ... base class declarations
   if (cl.BaseSize()) {
      for (Base_Iterator b = cl.Base_Begin(); b != cl.Base_End(); ++b) {
         generate_class_decl((*b).ToType(), indent);
      }
   }

   cout << indent << "class " << cl.Name();

   // ... bases
   if (cl.BaseSize()) {
      cout << " : ";

      for (Base_Iterator b = cl.Base_Begin(); b != cl.Base_End(); ++b) {
         if ((*b).IsVirtual()) {
            cout << "virtual ";
         }

         if ((*b).IsPublic()) {
            cout << "public ";
         }

         if ((*b).IsProtected()) {
            cout << "protected ";
         }

         if ((*b).IsPrivate()) {
            cout << "private ";
         }

         cout << (*b).ToType().Name(SCOPED);

         if (b != cl.Base_End() - 1) {
            cout << ", ";
         }
      }
   }

   cout << " {" << endl;

   Visibility vis = Private;

   // ... function members
   for (Member_Iterator f = cl.FunctionMember_Begin(); f != cl.FunctionMember_End(); ++f) {
      if (!(*f).IsArtificial()) {
         if ((*f).IsPublic() && vis != Public) {
            cout << indent << "public:" << endl;
            vis = Public;
         } else if ((*f).IsProtected() && vis != Protected) {
            cout << indent << "protected:" << endl;
            vis = Protected;
         } else if ((*f).IsPrivate() && vis != Private) {
            cout << indent << "private:" << endl;
            vis = Private;
         }

         Type ft = (*f).TypeOf();

         cout << indent + "  ";

         if (!(*f).IsConstructor() && !(*f).IsDestructor()) {
            cout << ft.ReturnType().Name(SCOPED) << " ";
         }

         if ((*f).IsOperator()) {
            cout << "operator ";
         }
         cout << (*f).Name() << " (";

         if (ft.FunctionParameterSize()) {
            for (size_t p = 0; p < ft.FunctionParameterSize(); p++) {
               cout << ft.FunctionParameterAt(p).Name(SCOPED | QUALIFIED);

               if ((*f).FunctionParameterNameAt(p).length()) {
                  cout << " " << (*f).FunctionParameterNameAt(p);
               }

               if ((*f).FunctionParameterDefaultAt(p).length()) {
                  cout << " = " << (*f).FunctionParameterDefaultAt(p);
               }

               if (p != ft.FunctionParameterSize() - 1) {
                  cout << ", ";
               }
            }
         }
         cout << ");" << endl;
      }
   }

   // ... data members
   for (Member_Iterator d = cl.DataMember_Begin(); d != cl.DataMember_End(); ++d) {
      if ((*d).IsPublic() && vis != Public) {
         cout << indent << "public:" << endl;
         vis = Public;
      } else if ((*d).IsProtected() && vis != Protected) {
         cout << indent << "protected:" << endl;
         vis = Protected;
      } else if ((*d).IsPrivate() && vis != Private) {
         cout << indent << "private:" << endl;
         vis = Private;
      }
      cout << indent + "  " << (*d).TypeOf().Name(SCOPED)
           << " " << (*d).Name() << ";" << endl;
   }
   cout << indent << "};" << endl;
} // generate_class_decl


void
generate_class(const Type& ty) {
   std::string indent = "";
   Scope sc = ty.DeclaringScope();

   // ... declaring scope
   if (!sc.IsTopScope()) {
      if (sc.IsNamespace()) {
         cout << "namespace ";
      } else if (sc.IsClass()) {
         cout << "class ";
      }

      cout << sc.Name() << " {" << endl;
      indent += "  ";
   }

   generate_class_decl(ty, indent);

   if (!sc.IsTopScope()) {
      cout << "}" << endl;

      if (sc.IsClass()) {
         cout << ";";
      }
   }
} // generate_class


using namespace ROOT::Reflex;

/**
 * test_Reflex_simple2.cpp
 * testing Reflex with a simple test dictionary
 */

class ReflexSimple2Test: public CppUnit::TestFixture {
   CPPUNIT_TEST_SUITE(ReflexSimple2Test);
   CPPUNIT_TEST(loadLibrary);
   CPPUNIT_TEST(testTemplateClass);
   CPPUNIT_TEST(testTemplatedMemberTypes);
   CPPUNIT_TEST(testIterators);
   CPPUNIT_TEST(fooBarZot);
   CPPUNIT_TEST(testBaseClasses);
   CPPUNIT_TEST(testDataMembers);
   CPPUNIT_TEST(testFunctionMembers);
   CPPUNIT_TEST(testFreeFunctions);
   CPPUNIT_TEST(testReferenceDataMembers);
   CPPUNIT_TEST(testReferenceArgs);
   CPPUNIT_TEST(testDiamond);
   CPPUNIT_TEST(testOperators);
   CPPUNIT_TEST(testTypedefSelection);
   CPPUNIT_TEST(testTypedef);
   CPPUNIT_TEST(testCppSelection);
   CPPUNIT_TEST(testCppSelectNoAutoselect);
   CPPUNIT_TEST(testTypedefInClass);
   CPPUNIT_TEST(testConstMembers);
   CPPUNIT_TEST(testSubTypes);
   CPPUNIT_TEST(testToTypeFinal);
   CPPUNIT_TEST(testScopeSubFuns);
   CPPUNIT_TEST(testEnums);
   CPPUNIT_TEST(fundamentalType);
   CPPUNIT_TEST(unloadType);
   CPPUNIT_TEST(iterateVector);
   CPPUNIT_TEST(testClasses);
   CPPUNIT_TEST(testTemplateTypedefs);
   CPPUNIT_TEST(testArray);
   CPPUNIT_TEST(testCommentsEtc);

   CPPUNIT_TEST(unloadLibrary);
   CPPUNIT_TEST(shutdown);
   CPPUNIT_TEST_SUITE_END();

public:
   void
   setUp() {}

   void loadLibrary();
   void testTemplateClass();
   void testTemplatedMemberTypes();
   void testIterators();
   void fooBarZot();
   void testBaseClasses();
   void testDataMembers();
   void testFunctionMembers();
   void testFreeFunctions();
   void testReferenceDataMembers();
   void testReferenceArgs();
   void testDiamond();
   void testOperators();
   void testTypedefSelection();
   void testTypedef();
   void testCppSelection();
   void testCppSelectNoAutoselect();
   void testTypedefInClass();
   void testConstMembers();
   void testSubTypes();
   void testToTypeFinal();
   void testScopeSubFuns();
   void testEnums();
   void fundamentalType();
   void unloadType();
   void iterateVector();
   void testClasses();
   void testTemplateTypedefs();
   void testArray();
   void testCommentsEtc();

   void unloadLibrary();
   void
   shutdown() {}

   void
   tearDown() {}

private:
   void testReferenceArgs_T(const std::string& Tname);
   void testReferenceDataMembers_T(const std::string& Tname);

}; // class ReflexSimple2Test

#if defined(_WIN32)
static HMODULE s_libInstance = 0;
#else
static void* s_libInstance = 0;
#endif

// loading the dictionary library
void
ReflexSimple2Test::loadLibrary() {
   //Reflex::accessArtificialMembers() = true;
#if defined(_WIN32)
   s_libInstance = LoadLibrary("libtest_Class2DictRflx.dll");
#else
   s_libInstance = dlopen("libtest_Class2DictRflx.so", RTLD_NOW);
#endif
   CPPUNIT_ASSERT(s_libInstance);
}


void
ReflexSimple2Test::testTemplateClass() {
   Type t = Type::ByName("TT::Outer<TT::A<unsigned long> >");
   CPPUNIT_ASSERT(t);
   CPPUNIT_ASSERT_EQUAL(size_t(2), ((Scope) t).SubScopeLevel());
   int numFuns = 0;

   for (Member_Iterator mi = t.FunctionMember_Begin(); mi != t.FunctionMember_End(); ++mi) {
      if (!(*mi).IsArtificial()) {
         ++numFuns;
      }
   }
   CPPUNIT_ASSERT_EQUAL(1, numFuns);
}


void
ReflexSimple2Test::testTemplatedMemberTypes() {
   Type t = Type::ByName("TT::TemplatedMemberTypes");
   CPPUNIT_ASSERT(t);

   Member m;
   Type tt;

   m = t.MemberByName("m0");
   CPPUNIT_ASSERT(m);
   tt = m.TypeOf();
   CPPUNIT_ASSERT(tt);
   CPPUNIT_ASSERT_EQUAL(std::string("std"), tt.DeclaringScope().Name(SCOPED));
   CPPUNIT_ASSERT_EQUAL(std::string("std"), Tools::GetScopeName(tt.Name(SCOPED | QUALIFIED)));

   m = t.MemberByName("m1");
   CPPUNIT_ASSERT(m);
   tt = m.TypeOf();
   CPPUNIT_ASSERT_EQUAL(std::string("std"), Tools::GetScopeName(tt.Name(SCOPED | QUALIFIED)));
   CPPUNIT_ASSERT(tt);
   CPPUNIT_ASSERT(tt.IsPointer());
   tt = tt.ToType();
   CPPUNIT_ASSERT(tt);
   CPPUNIT_ASSERT(tt.IsTypedef());
   CPPUNIT_ASSERT_EQUAL(std::string("std"), tt.DeclaringScope().Name(SCOPED));

   // FIXME: references are not yet supported
   m = t.MemberByName("m2");
   CPPUNIT_ASSERT(!m);

   /*
      tt = m.TypeOf();
      CPPUNIT_ASSERT_EQUAL("std", Tools::GetScopeName(tt.Name(SCOPED|QUALIFIED)));
      CPPUNIT_ASSERT(tt);
      CPPUNIT_ASSERT(tt.IsPointer());
      tt = tt.ToType();
      CPPUNIT_ASSERT(tt);
      CPPUNIT_ASSERT(tt.IsTypedef());
      CPPUNIT_ASSERT(tt.IsClass());
      CPPUNIT_ASSERT_EQUAL(std::string("std"),tt.DeclaringScope().Name(SCOPED));
    */

   m = t.MemberByName("m3");
   CPPUNIT_ASSERT(m);
   tt = m.TypeOf();
   CPPUNIT_ASSERT_EQUAL(std::string("std"), Tools::GetScopeName(tt.Name(SCOPED | QUALIFIED)));
   CPPUNIT_ASSERT(tt);
   CPPUNIT_ASSERT(tt.IsArray());
   CPPUNIT_ASSERT_EQUAL(5, int (tt.ArrayLength()));
   tt = tt.ToType();
   CPPUNIT_ASSERT(tt);
   CPPUNIT_ASSERT(tt.IsTypedef());
   CPPUNIT_ASSERT_EQUAL(std::string("std"), tt.DeclaringScope().Name(SCOPED));

   m = t.MemberByName("m4");
   CPPUNIT_ASSERT(m);
   tt = m.TypeOf();
   CPPUNIT_ASSERT_EQUAL(std::string("std"), Tools::GetScopeName(tt.Name(SCOPED | QUALIFIED)));
   CPPUNIT_ASSERT(tt);
   CPPUNIT_ASSERT(tt.IsPointer());
   tt = tt.ToType();
   CPPUNIT_ASSERT(tt);
   CPPUNIT_ASSERT(tt.IsTemplateInstance());
   CPPUNIT_ASSERT_EQUAL(std::string("std"), tt.DeclaringScope().Name(SCOPED));

   m = t.MemberByName("m5");
   CPPUNIT_ASSERT(m);
   tt = m.TypeOf();
   CPPUNIT_ASSERT_EQUAL(std::string("std"), Tools::GetScopeName(tt.Name(SCOPED | QUALIFIED)));
   CPPUNIT_ASSERT(tt);
   CPPUNIT_ASSERT(tt.IsArray());
   CPPUNIT_ASSERT_EQUAL(5, int (tt.ArrayLength()));
   tt = tt.ToType();
   CPPUNIT_ASSERT(tt);
   CPPUNIT_ASSERT(tt.IsClass());
   CPPUNIT_ASSERT_EQUAL(std::string("std"), tt.DeclaringScope().Name(SCOPED));

} // testTemplatedMemberTypes


void
ReflexSimple2Test::testIterators() {
   CPPUNIT_ASSERT_EQUAL(Scope::Scope_Begin()->Name(), (Scope::Scope_REnd() - 1)->Name());
   CPPUNIT_ASSERT_EQUAL((Scope::Scope_End() - 1)->Name(), Scope::Scope_RBegin()->Name());
   CPPUNIT_ASSERT_EQUAL(Type::Type_Begin()->Name(), (Type::Type_REnd() - 1)->Name());
   CPPUNIT_ASSERT_EQUAL((Type::Type_End() - 1)->Name(), Type::Type_RBegin()->Name());

   Scope s = Scope::ByName("");
   CPPUNIT_ASSERT(s);
   CPPUNIT_ASSERT(s.Id());
   CPPUNIT_ASSERT(s.IsTopScope());

   if (s.SubScopeSize()) {
      CPPUNIT_ASSERT_EQUAL(s.SubScope_Begin()->Name(), (s.SubScope_REnd() - 1)->Name());
      CPPUNIT_ASSERT_EQUAL(s.SubScope_RBegin()->Name(), (s.SubScope_End() - 1)->Name());
   }

   if (s.SubTypeSize()) {
      CPPUNIT_ASSERT_EQUAL(s.SubType_Begin()->Name(), (s.SubType_REnd() - 1)->Name());
      CPPUNIT_ASSERT_EQUAL(s.SubType_RBegin()->Name(), (s.SubType_End() - 1)->Name());
   }

   if (s.SubTypeTemplateSize()) {
      CPPUNIT_ASSERT_EQUAL(s.SubTypeTemplate_Begin()->Name(), (s.SubTypeTemplate_REnd() - 1)->Name());
      CPPUNIT_ASSERT_EQUAL(s.SubTypeTemplate_RBegin()->Name(), (s.SubTypeTemplate_End() - 1)->Name());
   }
   Scope s2 = Scope::ByName("ClassF");
   CPPUNIT_ASSERT(s2);
   CPPUNIT_ASSERT(s2.Id());

   if (s2.BaseSize()) {
      CPPUNIT_ASSERT_EQUAL(s2.Base_Begin()->Name(), (s2.Base_REnd() - 1)->Name());
      CPPUNIT_ASSERT_EQUAL(s2.Base_RBegin()->Name(), (s2.Base_End() - 1)->Name());
   }

   if (s2.DataMemberSize()) {
      CPPUNIT_ASSERT_EQUAL(s2.DataMember_Begin()->Name(), (s2.DataMember_REnd() - 1)->Name());
      CPPUNIT_ASSERT_EQUAL(s2.DataMember_RBegin()->Name(), (s2.DataMember_End() - 1)->Name());
   }

   if (s2.FunctionMemberSize()) {
      CPPUNIT_ASSERT_EQUAL(s2.FunctionMember_Begin()->Name(), (s2.FunctionMember_REnd() - 1)->Name());
      CPPUNIT_ASSERT_EQUAL(s2.FunctionMember_RBegin()->Name(), (s2.FunctionMember_End() - 1)->Name());
   }

   if (s2.MemberSize()) {
      CPPUNIT_ASSERT_EQUAL(s2.Member_Begin()->Name(), (s2.Member_REnd() - 1)->Name());
      CPPUNIT_ASSERT_EQUAL(s2.Member_RBegin()->Name(), (s2.Member_End() - 1)->Name());
   }

   if (s2.MemberTemplateSize()) {
      CPPUNIT_ASSERT_EQUAL(s2.MemberTemplate_Begin()->Name(), (s2.MemberTemplate_REnd() - 1)->Name());
      CPPUNIT_ASSERT_EQUAL(s2.MemberTemplate_RBegin()->Name(), (s2.MemberTemplate_End() - 1)->Name());
   }

   if (s2.TemplateArgumentSize()) {
      CPPUNIT_ASSERT_EQUAL(s2.TemplateArgument_Begin()->Name(), (s2.TemplateArgument_REnd() - 1)->Name());
      CPPUNIT_ASSERT_EQUAL(s2.TemplateArgument_RBegin()->Name(), (s2.TemplateArgument_End() - 1)->Name());
   }
} // testIterators


void
ReflexSimple2Test::fooBarZot() {
   // get meta information for class foo
   Type t = Type::ByName("zot::foo");
   CPPUNIT_ASSERT(t);
   // generate declarations for foo
   //if (t) generate_class(t);


   // get meta information of type Foo
   Type fooType = Type::ByName("zot::foo");
   CPPUNIT_ASSERT(fooType);

   // check if the type is valid
   if (fooType) {
      //
      // Introspection
      //

      // get the name of the foo type (i.e. "Foo")
      std::string fooName = fooType.Name();
      CPPUNIT_ASSERT_EQUAL(std::string("foo"), fooName);

      // get number of base classes (i.e. 1)
      size_t fooBases = fooType.BaseSize();
      CPPUNIT_ASSERT_EQUAL(size_t(1), fooBases);
      // get first base class information
      Base fooBase = fooType.BaseAt(0);
      CPPUNIT_ASSERT(fooBase);
      // get name of first base class (i.e. "FooBase")
      std::string fooBaseName = fooBase.Name();
      CPPUNIT_ASSERT_EQUAL(std::string("foo_base"), fooBaseName);
      // check virtual inheritance (i.e. true)
      bool inheritsVirtual = fooBase.IsVirtual();
      CPPUNIT_ASSERT_EQUAL(inheritsVirtual, true);
      // check if publically inherited (i.e. true)
      bool inheritsPublic = fooBase.IsPublic();
      CPPUNIT_ASSERT_EQUAL(inheritsPublic, true);

      // get number of members (i.e. 15)
      fooType.UpdateMembers();
      size_t fooMembers = fooType.MemberSize();
      CPPUNIT_ASSERT_EQUAL(size_t(15), fooMembers);

      // get number of data members (i.e. 1)
      size_t fooDataMembers = fooType.DataMemberSize();
      CPPUNIT_ASSERT_EQUAL(size_t(1), fooDataMembers);
      // retrieve data member "fBar"
      Member dm = fooType.DataMemberByName("fBar");
      CPPUNIT_ASSERT(dm);
      // retrieve the type of this data member
      Type dmType = dm.TypeOf();
      CPPUNIT_ASSERT(dmType);
      // name of type of the data member (i.e. "int")
      std::string dmTypeName = dmType.Name();
      CPPUNIT_ASSERT_EQUAL(std::string("int"), dmTypeName);

      // get the function member "bar"
      Member fm = fooType.FunctionMemberByName("bar");
      CPPUNIT_ASSERT(fm);
      // name of the function member (i.e. "bar")
      std::string fmName = fm.Name();
      CPPUNIT_ASSERT_EQUAL(std::string("bar"), fmName);
      // name of type of the function member (i.e. "int (void)")
      std::string fmTypeName = fm.TypeOf().Name();
      CPPUNIT_ASSERT_EQUAL(std::string("int (void)"), fmTypeName);
      // name of return type of function member (i.e. "int")
      std::string fmReturnTypeName = fm.TypeOf().ReturnType().Name();
      CPPUNIT_ASSERT_EQUAL(std::string("int"), fmReturnTypeName);

      //
      // Interaction
      //

      // update the information for inherited members of class foo
      fooType.UpdateMembers();

      // construct an object of type Foo
      Object fooObj = fooType.Construct();
      CPPUNIT_ASSERT(fooObj);

      // get the value of the data member (i.e. 4711)
      int val = Object_Cast<int>(fooObj.Get("fBar"));
      CPPUNIT_ASSERT_EQUAL(4711, val);

      // set the data member to 4712
      fooObj.Set("fBar", ++val);
      // get the data member again (i.e. 4712)
      val = Object_Cast<int>(fooObj.Get("fBar"));
      CPPUNIT_ASSERT_EQUAL(4712, val);

      // call function setBar with value 4713
      std::vector<void*> vec;
      vec.push_back(&(++val));
      fooObj.Invoke("set_bar", Type::ByName("void (int)"), 0, vec);
      // call operator ++ to increase fBar by one
      fooObj.Invoke("operator++");
      // call bar getter and cast the output to int (i.e. 4714)
      fooObj.Invoke("bar", val);
      CPPUNIT_ASSERT_EQUAL(4714, val);

      // delete the Foo object
      fooObj.Destruct();
      CPPUNIT_ASSERT(!fooObj.Address());
      CPPUNIT_ASSERT(!fooObj);
   }

} // fooBarZot


// testing base classes
void
ReflexSimple2Test::testBaseClasses() {
   Type t1 = Type::ByName("ClassH");
   Type t2 = Type::ByName("ClassB");
   CPPUNIT_ASSERT(t1);
   CPPUNIT_ASSERT_EQUAL(size_t(1), ((Scope) t1).SubScopeLevel());
   CPPUNIT_ASSERT(t2);
   CPPUNIT_ASSERT(t1.HasBase(t2));
}


// testing data members
void
ReflexSimple2Test::testDataMembers() {
   Type t1;
   Object o1;

   t1 = Type::ByName("ClassH");
   CPPUNIT_ASSERT(t1);
   o1 = t1.Construct();
   CPPUNIT_ASSERT(o1);
   CPPUNIT_ASSERT_EQUAL(int (t1.DataMemberSize()), 1);
   CPPUNIT_ASSERT_EQUAL(std::string("ClassH::fH"), t1.DataMemberAt(0).Name(S));
   t1.UpdateMembers();
   CPPUNIT_ASSERT_EQUAL(int (t1.DataMemberSize()), 9);
   CPPUNIT_ASSERT_EQUAL(std::string("ClassH::fH"), t1.DataMemberAt(0).Name(S));
   CPPUNIT_ASSERT_EQUAL(std::string("ClassG::fG"), t1.DataMemberAt(1).Name(S));
   CPPUNIT_ASSERT_EQUAL(std::string("ClassF::fF"), t1.DataMemberAt(2).Name(S));
   CPPUNIT_ASSERT_EQUAL(std::string("ClassD::fD"), t1.DataMemberAt(3).Name(S));
   CPPUNIT_ASSERT_EQUAL(std::string("ClassB::fB"), t1.DataMemberAt(4).Name(S));
   CPPUNIT_ASSERT_EQUAL(std::string("ClassA::fA"), t1.DataMemberAt(5).Name(S));
   CPPUNIT_ASSERT_EQUAL(std::string("ClassM::fM"), t1.DataMemberAt(6).Name(S));
   CPPUNIT_ASSERT_EQUAL(std::string("ClassE::fE"), t1.DataMemberAt(7).Name(S));
   CPPUNIT_ASSERT_EQUAL(std::string("ClassC::fC"), t1.DataMemberAt(8).Name(S));
   CPPUNIT_ASSERT_EQUAL(std::string("private fC"), t1.DataMemberAt(8).Name(Q));

   Member m1;
   char val;
   int ii;

   m1 = t1.DataMemberAt(0);
   CPPUNIT_ASSERT(m1);
   val = (char) *(int*) m1.Get(o1).Address();
   CPPUNIT_ASSERT_EQUAL('h', val);
   ++val;
   m1.Set(o1, (void*) &(ii = (int) val));
   val = (char) *(int*) m1.Get(o1).Address();
   CPPUNIT_ASSERT_EQUAL('i', val);

   m1 = t1.DataMemberAt(1);
   CPPUNIT_ASSERT(m1);
   val = (char) *(int*) m1.Get(o1).Address();
   CPPUNIT_ASSERT_EQUAL('g', val);
   ++val;
   m1.Set(o1, (void*) &(ii = (int) val));
   val = (char) *(int*) m1.Get(o1).Address();
   CPPUNIT_ASSERT_EQUAL('h', val);

   m1 = t1.DataMemberAt(2);
   CPPUNIT_ASSERT(m1);
   val = (char) *(int*) m1.Get(o1).Address();
   CPPUNIT_ASSERT_EQUAL('f', val);
   ++val;
   m1.Set(o1, (void*) &(ii = (int) val));
   val = (char) *(int*) m1.Get(o1).Address();
   CPPUNIT_ASSERT_EQUAL('g', val);

   m1 = t1.DataMemberAt(3);
   CPPUNIT_ASSERT(m1);
   val = (char) *(int*) m1.Get(o1).Address();
   CPPUNIT_ASSERT_EQUAL('d', val);
   ++val;
   m1.Set(o1, (void*) &(ii = (int) val));
   val = (char) *(int*) m1.Get(o1).Address();
   CPPUNIT_ASSERT_EQUAL('e', val);

   m1 = t1.DataMemberAt(4);
   CPPUNIT_ASSERT(m1);
   val = (char) *(int*) m1.Get(o1).Address();
   CPPUNIT_ASSERT_EQUAL('b', val);
   ++val;
   m1.Set(o1, (void*) &(ii = (int) val));
   val = (char) *(int*) m1.Get(o1).Address();
   CPPUNIT_ASSERT_EQUAL('c', val);

   m1 = t1.DataMemberAt(5);
   CPPUNIT_ASSERT(m1);
   val = (char) *(int*) m1.Get(o1).Address();
   CPPUNIT_ASSERT_EQUAL('a', val);
   ++val;
   m1.Set(o1, (void*) &(ii = (int) val));
   val = (char) *(int*) m1.Get(o1).Address();
   CPPUNIT_ASSERT_EQUAL('b', val);

   m1 = t1.DataMemberAt(6);
   CPPUNIT_ASSERT(m1);
   val = (char) *(int*) m1.Get(o1).Address();
   CPPUNIT_ASSERT_EQUAL('m', val);
   ++val;
   m1.Set(o1, (void*) &(ii = (int) val));
   val = (char) *(int*) m1.Get(o1).Address();
   CPPUNIT_ASSERT_EQUAL('n', val);

   m1 = t1.DataMemberAt(7);
   CPPUNIT_ASSERT(m1);
   val = (char) *(int*) m1.Get(o1).Address();
   CPPUNIT_ASSERT_EQUAL('e', val);
   ++val;
   m1.Set(o1, (void*) &(ii = (int) val));
   val = (char) *(int*) m1.Get(o1).Address();
   CPPUNIT_ASSERT_EQUAL('f', val);

   m1 = t1.DataMemberAt(8);
   CPPUNIT_ASSERT(m1);
   val = (char) *(int*) m1.Get(o1).Address();
   CPPUNIT_ASSERT_EQUAL('c', val);
   ++val;
   m1.Set(o1, (void*) &(ii = (int) val));
   val = (char) *(int*) m1.Get(o1).Address();
   CPPUNIT_ASSERT_EQUAL('d', val);

   Type t2 = Type::ByName("testclasses::DataMembers");
   CPPUNIT_ASSERT(t2);
   CPPUNIT_ASSERT(t2.IsClass());

   Member m20 = t2.MemberByName("i");
   CPPUNIT_ASSERT(m20);
   Type m20t = m20.TypeOf();
   CPPUNIT_ASSERT(m20t);
   CPPUNIT_ASSERT_EQUAL(std::string(""), m20t.ToType().Name());
   CPPUNIT_ASSERT_EQUAL(m20t.Name(), m20t.FinalType().Name());
   CPPUNIT_ASSERT_EQUAL(m20t.Name(), m20t.RawType().Name());

   Member m21 = t2.MemberByName("pi");
   CPPUNIT_ASSERT(m21);
   Type m21t = m21.TypeOf();
   CPPUNIT_ASSERT(m21t);
   CPPUNIT_ASSERT_EQUAL(m21t.Name(), m21t.FinalType().Name());
   CPPUNIT_ASSERT_EQUAL(m20t.Name(), m21t.RawType().Name());

   Member m22 = t2.MemberByName("ppi");
   CPPUNIT_ASSERT(m22);
   Type m22t = m22.TypeOf();
   CPPUNIT_ASSERT(m22t);
   CPPUNIT_ASSERT_EQUAL(m22t.Name(), m22t.FinalType().Name());
   CPPUNIT_ASSERT_EQUAL(m20t.Name(), m22t.RawType().Name());
   CPPUNIT_ASSERT_EQUAL(m20t.Name(), m22t.RawType().Name());
   CPPUNIT_ASSERT_EQUAL(m21t.Name(), m22t.FinalType().ToType().Name());
   CPPUNIT_ASSERT_EQUAL(std::string(""), m22t.RawType().ToType().Name());

   Member m23 = t2.MemberByName("pa");
   CPPUNIT_ASSERT(m23);
   Type m23t = m23.TypeOf();
   CPPUNIT_ASSERT(m23t);
   CPPUNIT_ASSERT_EQUAL(m23t.Name(), m23t.FinalType().Name());
   CPPUNIT_ASSERT_EQUAL(m20t.Name(), m23t.RawType().Name());
   CPPUNIT_ASSERT_EQUAL(m20t.Name(), m23t.RawType().Name());

   Member m24 = t2.MemberByName("paa");
   CPPUNIT_ASSERT(m24);
   Type m24t = m24.TypeOf();
   CPPUNIT_ASSERT(m24t);
   CPPUNIT_ASSERT_EQUAL(m24t.Name(), m24t.FinalType().Name());
   CPPUNIT_ASSERT_EQUAL(m20t.Name(), m24t.RawType().Name());
   CPPUNIT_ASSERT_EQUAL(m20t.Name(), m24t.RawType().Name());

   o1.Destruct();
} // testDataMembers


void
ReflexSimple2Test::testFunctionMembers() {
   Type t;
   Scope s;
   Object o;
   Member m;

   t = Type::ByName("ClassH");
   CPPUNIT_ASSERT(t);

   o = t.Construct();
   CPPUNIT_ASSERT(o);

   CPPUNIT_ASSERT_EQUAL(61, int (t.FunctionMemberSize()));

   m = t.MemberByName("h");
   CPPUNIT_ASSERT(m);
   CPPUNIT_ASSERT(m.DeclaringType());
   CPPUNIT_ASSERT_EQUAL(std::string("ClassH"), m.DeclaringType().Name());
   CPPUNIT_ASSERT(m.DeclaringScope());
   CPPUNIT_ASSERT_EQUAL(std::string("ClassH"), m.DeclaringScope().Name());
   CPPUNIT_ASSERT(m.DeclaringType() == (Type) m.DeclaringScope());
   char c = '?';
   m.Invoke(o, c);
   CPPUNIT_ASSERT_EQUAL('h', c);

   m = t.MemberByName("g");
   CPPUNIT_ASSERT(m);
   m.Invoke(o, c);
   CPPUNIT_ASSERT_EQUAL('g', c);


   m = t.MemberByName("setG");
   CPPUNIT_ASSERT(m);
   CPPUNIT_ASSERT_EQUAL(size_t(0), m.FunctionParameterSize(true));
   CPPUNIT_ASSERT_EQUAL(size_t(1), m.FunctionParameterSize());
   CPPUNIT_ASSERT_EQUAL(std::string("v"), *m.FunctionParameterName_Begin());
   CPPUNIT_ASSERT_EQUAL(*m.FunctionParameterName_Begin(), *(m.FunctionParameterName_REnd() - 1));
   CPPUNIT_ASSERT_EQUAL(*m.FunctionParameterName_RBegin(), *(m.FunctionParameterName_End() - 1));
   CPPUNIT_ASSERT_EQUAL(std::string("11"), *m.FunctionParameterDefault_Begin());
   CPPUNIT_ASSERT_EQUAL(*m.FunctionParameterDefault_Begin(), *(m.FunctionParameterDefault_REnd() - 1));
   CPPUNIT_ASSERT_EQUAL(*m.FunctionParameterDefault_RBegin(), *(m.FunctionParameterDefault_End() - 1));
   CPPUNIT_ASSERT_EQUAL(std::string("11"), *(m.FunctionParameterDefault_RBegin()));
   CPPUNIT_ASSERT_EQUAL(std::string("11"), *(m.FunctionParameterDefault_REnd() - 1));
   CPPUNIT_ASSERT_EQUAL(typeid(ROOT::Reflex::StubFunction).name(), typeid(m.Stubfunction()).name());

   m = t.MemberByName("f");
   CPPUNIT_ASSERT(m);
   m.Invoke(o, c);
   CPPUNIT_ASSERT_EQUAL('f', c);

   m = t.MemberByName("d");
   CPPUNIT_ASSERT(m);
   m.Invoke(o, c);
   CPPUNIT_ASSERT_EQUAL('d', c);

   m = t.MemberByName("b");
   CPPUNIT_ASSERT(m);
   m.Invoke(o, c);
   CPPUNIT_ASSERT_EQUAL('b', c);

   m = t.MemberByName("a");
   CPPUNIT_ASSERT(m);
   m.Invoke(o, c);
   CPPUNIT_ASSERT_EQUAL('a', c);

   m = t.MemberByName("m");
   CPPUNIT_ASSERT(m);
   m.Invoke(o, c);
   CPPUNIT_ASSERT_EQUAL('m', c);

   m = t.MemberByName("e");
   CPPUNIT_ASSERT(m);
   m.Invoke(o, c);
   CPPUNIT_ASSERT_EQUAL('e', c);

   m = t.MemberByName("c");
   CPPUNIT_ASSERT(m);
   m.Invoke(o, c);
   CPPUNIT_ASSERT_EQUAL('c', c);

   o.Destruct();

} // testFunctionMembers


void
ReflexSimple2Test::testFreeFunctions() {
   Scope s;
   Member m;
   Type t;
   std::vector<void*> vec;

   s = Scope::ByName("Functions");
   CPPUNIT_ASSERT(s);
   CPPUNIT_ASSERT_EQUAL(4, int (s.FunctionMemberSize()));

   int i = 1;
   vec.push_back((void*) &i);
   m = s.FunctionMemberByName("function4");
   CPPUNIT_ASSERT(m);
   CPPUNIT_ASSERT_EQUAL(std::string("function4"), m.Name());
   CPPUNIT_ASSERT_EQUAL(std::string("int (int)"), m.TypeOf().Name());
   int ret = -42;
   m.Invoke(Object(), ret, vec);
   CPPUNIT_ASSERT_EQUAL(11, ret);

   float f = 1.0;
   vec.push_back((void*) &f);
   m = s.FunctionMemberByName("function3");
   CPPUNIT_ASSERT(m);
   CPPUNIT_ASSERT_EQUAL(std::string("function3"), m.Name());
   CPPUNIT_ASSERT_EQUAL(std::string("double* (int, float)"), m.TypeOf().Name());
   double* pret_d = 0;
   m.Invoke(Object(), pret_d, vec);
   CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0, *pret_d, 0);

   m = s.FunctionMemberByName("function2");
   CPPUNIT_ASSERT(m);
   CPPUNIT_ASSERT_EQUAL(std::string("function2"), m.Name());
   CPPUNIT_ASSERT_EQUAL(std::string("int (void)"), m.TypeOf().Name());
   m.Invoke(Object(), ret, std::vector<void*>());
   CPPUNIT_ASSERT_EQUAL(999, ret);

   m = s.FunctionMemberByName("function1");
   CPPUNIT_ASSERT(m);
   CPPUNIT_ASSERT_EQUAL(std::string("function1"), m.Name());
   CPPUNIT_ASSERT_EQUAL(std::string("void (void)"), m.TypeOf().Name());
   Object ro;
   m.Invoke(Object(), &ro, std::vector<void*>());
   CPPUNIT_ASSERT(!ro);

   t = Type::ByName("ClassAAA");
   CPPUNIT_ASSERT(t);
   CPPUNIT_ASSERT_EQUAL(6, int (t.MemberSize()));
   CPPUNIT_ASSERT_EQUAL(6, int (t.FunctionMemberSize()));
   CPPUNIT_ASSERT_EQUAL(0, int (t.DataMemberSize()));
   m = t.MemberByName("function6");
   CPPUNIT_ASSERT(m);
   CPPUNIT_ASSERT_EQUAL(std::string("function6"), m.Name());
   CPPUNIT_ASSERT_EQUAL(std::string("int (int)"), m.TypeOf().Name());

   s = t.DeclaringScope();
   CPPUNIT_ASSERT(s);
   CPPUNIT_ASSERT(s.IsTopScope());
   CPPUNIT_ASSERT_EQUAL(1, int (s.DataMemberSize()));
   CPPUNIT_ASSERT_EQUAL(8, int (s.FunctionMemberSize()));
   m = s.MemberByName("function5");
   CPPUNIT_ASSERT(m);
   CPPUNIT_ASSERT_EQUAL(std::string("function5"), m.Name());
   CPPUNIT_ASSERT_EQUAL(std::string("function5"), m.Name(SCOPED));
   CPPUNIT_ASSERT_EQUAL(std::string("int (MYINT)"), m.TypeOf().Name());
   CPPUNIT_ASSERT_EQUAL(std::string("int (int)"), m.TypeOf().Name(FINAL));
   CPPUNIT_ASSERT_EQUAL(std::string("int (int)"), m.TypeOf().Name(SCOPED | FINAL));

   t = Type::ByName("ClassBBB");
   CPPUNIT_ASSERT(t);
   CPPUNIT_ASSERT_EQUAL(6, int (t.MemberSize()));
   m = t.MemberByName("meth");
   CPPUNIT_ASSERT(m);
   CPPUNIT_ASSERT_EQUAL(std::string("ClassBBB::meth"), m.Name(SCOPED));
   CPPUNIT_ASSERT_EQUAL(std::string("int (int)"), m.TypeOf().Name());
   CPPUNIT_ASSERT_EQUAL(std::string("int (int)"), m.TypeOf().Name(SCOPED));
   CPPUNIT_ASSERT_EQUAL(std::string("int (int)"), m.TypeOf().Name(FINAL));
   CPPUNIT_ASSERT_EQUAL(std::string("int (int)"), m.TypeOf().Name(QUALIFIED));
   CPPUNIT_ASSERT_EQUAL(std::string("int (int)"), m.TypeOf().Name(SCOPED | FINAL));
   CPPUNIT_ASSERT_EQUAL(std::string("int (int)"), m.TypeOf().Name(SCOPED | QUALIFIED));
   CPPUNIT_ASSERT_EQUAL(std::string("int (int)"), m.TypeOf().Name(SCOPED | QUALIFIED | FINAL));
   CPPUNIT_ASSERT_EQUAL(std::string("int (int)"), m.TypeOf().Name(S));
   CPPUNIT_ASSERT_EQUAL(std::string("int (int)"), m.TypeOf().Name(F));
   CPPUNIT_ASSERT_EQUAL(std::string("int (int)"), m.TypeOf().Name(Q));
   CPPUNIT_ASSERT_EQUAL(std::string("int (int)"), m.TypeOf().Name(S | F));
   CPPUNIT_ASSERT_EQUAL(std::string("int (int)"), m.TypeOf().Name(S | Q));
   CPPUNIT_ASSERT_EQUAL(std::string("int (int)"), m.TypeOf().Name(S | Q | F));
   CPPUNIT_ASSERT_EQUAL(std::string("int (int)"), m.TypeOf().Name(SCOPED | S));
   CPPUNIT_ASSERT_EQUAL(std::string("int (int)"), m.TypeOf().Name(FINAL | F));
   CPPUNIT_ASSERT_EQUAL(std::string("int (int)"), m.TypeOf().Name(QUALIFIED | Q));
   CPPUNIT_ASSERT_EQUAL(std::string("int (int)"), m.TypeOf().Name(SCOPED | QUALIFIED | FINAL | S | Q | F));

   t = Type::ByName("ClassB");
   CPPUNIT_ASSERT(t);
   Object o = t.Construct();
   CPPUNIT_ASSERT(o);
   int arg = 2;
   std::vector<void*> argVec;

   for (int j = 0; j < 20; ++j) {
      argVec.push_back(&arg);
   }
   o.Invoke("funWithManyArgs", ret, argVec);
   CPPUNIT_ASSERT_EQUAL(ret, 40);

   o.Destruct();

} // testFreeFunctions


void
ReflexSimple2Test::testReferenceArgs() {
   testReferenceArgs_T("int");
   testReferenceArgs_T("ClassO<int>");
   testReferenceArgs_T("int*");
   testReferenceArgs_T("ClassO<int*>*");
}


void
ReflexSimple2Test::testReferenceArgs_T(const std::string& Tname) {
   std::string className("ClassO<");
   className += Tname;
   char lastChar = Tname[Tname.length() - 1];

   if (lastChar == '>') {
      className += ' ';
   }
   bool nameIsPointer = lastChar == '*';

   std::string cTname;

   if (nameIsPointer) {
      cTname = Tname + " const";
   } else {
      cTname = "const ";
      cTname += Tname;
   }

   className += ">";
   cout << "Testing for " << Tname << "..." << endl;

   Type c = Type::ByName(className);
   CPPUNIT_ASSERT(c);
   Member m;

   cout << "Need to pass QUALIFIED or TypeOf().Name() will be wrong: " << endl
        << "  default is to strip modifiers (like ref): to be fixed" << endl
        << "  default is to strip modifiers in all levels, even for func args: to be fixed" << endl;

   ENTITY_HANDLING modName = QUALIFIED; // should work with "0", too!

   m = c.MemberByName("P");
   CPPUNIT_ASSERT(m);
   CPPUNIT_ASSERT_EQUAL(std::string("void (") + Tname + "*)", m.TypeOf().Name(modName));
   CPPUNIT_ASSERT_EQUAL(std::string("void (") + Tname + "*)", m.TypeOf().Name(QUALIFIED));
   m = c.MemberByName("R");
   CPPUNIT_ASSERT(m);
   CPPUNIT_ASSERT_EQUAL(std::string("void (") + Tname + "&)", m.TypeOf().Name(modName));
   CPPUNIT_ASSERT_EQUAL(std::string("void (") + Tname + "&)", m.TypeOf().Name(QUALIFIED));
   m = c.MemberByName("cP");
   CPPUNIT_ASSERT(m);
   CPPUNIT_ASSERT_EQUAL(std::string("void (") + cTname + "*)", m.TypeOf().Name(modName));
   CPPUNIT_ASSERT_EQUAL(std::string("void (") + cTname + "*)", m.TypeOf().Name(QUALIFIED));
   m = c.MemberByName("cR");
   CPPUNIT_ASSERT(m);
   CPPUNIT_ASSERT_EQUAL(std::string("void (") + cTname + "&)", m.TypeOf().Name(modName));
   CPPUNIT_ASSERT_EQUAL(std::string("void (") + cTname + "&)", m.TypeOf().Name(QUALIFIED));
   m = c.MemberByName("cPc");
   CPPUNIT_ASSERT(m);
   CPPUNIT_ASSERT_EQUAL(std::string("void (") + cTname + "* const)", m.TypeOf().Name(modName));
   CPPUNIT_ASSERT_EQUAL(std::string("void (") + cTname + "* const)", m.TypeOf().Name(QUALIFIED));

   cout << "Testing for " << Tname << ": done." << endl << endl;
} // testReferenceArgs_T


void
ReflexSimple2Test::testReferenceDataMembers() {
   testReferenceDataMembers_T("int");
   testReferenceDataMembers_T("ClassO<int>");
   testReferenceDataMembers_T("int*");
   testReferenceDataMembers_T("ClassO<int*>*");
}


void
ReflexSimple2Test::testReferenceDataMembers_T(const std::string& Tname) {
   std::string className("ClassO<");
   className += Tname;

   if (Tname[Tname.length() - 1] == '>') {
      className += ' ';
   }
   className += ">";
   cout << "Testing for " << Tname << "..." << endl;

   Type c = Type::ByName(className);
   CPPUNIT_ASSERT(c);
   Member m;

   // data
   m = c.MemberByName("_p");
   CPPUNIT_ASSERT(m);
   CPPUNIT_ASSERT_EQUAL(Tname + "*", m.TypeOf().Name(QUALIFIED));
   m = c.MemberByName("_pp");
   CPPUNIT_ASSERT(m);
   CPPUNIT_ASSERT_EQUAL(Tname + "**", m.TypeOf().Name(QUALIFIED));
   m = c.MemberByName("_cpp");
   CPPUNIT_ASSERT(m);

   if (Tname[Tname.length() - 1] == '*') {
      CPPUNIT_ASSERT_EQUAL(Tname + " const**", m.TypeOf().Name(QUALIFIED));
   } else {
      CPPUNIT_ASSERT_EQUAL(std::string("const ") + Tname + "**", m.TypeOf().Name(QUALIFIED));
   }

   cout << "Testing for " << Tname << ": done." << endl << endl;
} // testReferenceDataMembers_T


void
ReflexSimple2Test::testDiamond() {
   Type b = Type::ByName("Bla::Base");
   Type d = Type::ByName("Bla::Diamond");
   Type l = Type::ByName("Bla::Left");
   Type r = Type::ByName("Bla::Right");

   CPPUNIT_ASSERT(b);
   CPPUNIT_ASSERT(d);
   CPPUNIT_ASSERT(l);
   CPPUNIT_ASSERT(r);

   Type s = Type::ByName("void (void)");
   CPPUNIT_ASSERT(s);

   std::vector<void*> values;

   Member m = b.DataMemberAt(0);
   CPPUNIT_ASSERT(m);

   Object o = d.Construct(s, values);
   CPPUNIT_ASSERT(o);
   CPPUNIT_ASSERT(*(int*) m.Get(o).Address());
   o.Destruct();

   o = l.Construct(s, values);
   CPPUNIT_ASSERT_EQUAL(std::string("Base"), l.BaseAt(0).Name());
   CPPUNIT_ASSERT_EQUAL(std::string("public virtual Base"), l.BaseAt(0).Name(Q));
   CPPUNIT_ASSERT_EQUAL(99, *(int*) m.Get(o).Address());
   l.UpdateMembers();
   CPPUNIT_ASSERT_EQUAL(99, *(int*) m.Get(o).Address());
   o.Destruct();
} // testDiamond


int
countNewOperators(const Type& t) {
   int cnt = 0;

   for (Member_Iterator mi = t.FunctionMember_Begin(); mi != t.FunctionMember_End(); ++mi) {
      if ((*mi).IsOperator() && ((*mi).Name() == "operator new" || (*mi).Name() == "operator new []")) {
         ++cnt;
      }
   }
   return cnt;
}


void
ReflexSimple2Test::testOperators() {
   Type t1 = Type::ByName("testclasses::OverloadedOperators::NoOp");
   CPPUNIT_ASSERT(t1);
   CPPUNIT_ASSERT_EQUAL(0, countNewOperators(t1));
   CPPUNIT_ASSERT_EQUAL((size_t) 3, ((Scope) t1).SubScopeLevel());


   Type t2 = Type::ByName("testclasses::OverloadedOperators::OpNew");
   CPPUNIT_ASSERT(t2);
   CPPUNIT_ASSERT_EQUAL(1, countNewOperators(t2));

   Type t3 = Type::ByName("testclasses::OverloadedOperators::PlOpNew");
   CPPUNIT_ASSERT(t3);
   CPPUNIT_ASSERT_EQUAL(1, countNewOperators(t3));

   Type t4 = Type::ByName("testclasses::OverloadedOperators::PlOpOpNew");
   CPPUNIT_ASSERT(t4);
   CPPUNIT_ASSERT_EQUAL(2, countNewOperators(t4));

   Type t5 = Type::ByName("testclasses::OverloadedOperators::OpANew");
   CPPUNIT_ASSERT(t5);
   CPPUNIT_ASSERT_EQUAL(1, countNewOperators(t5));

   Type t6 = Type::ByName("testclasses::OverloadedOperators::PlOpANew");
   CPPUNIT_ASSERT(t6);
   CPPUNIT_ASSERT_EQUAL(1, countNewOperators(t6));

   Type t7 = Type::ByName("testclasses::OverloadedOperators::PlOpAOpANew");
   CPPUNIT_ASSERT(t7);
   CPPUNIT_ASSERT_EQUAL(2, countNewOperators(t7));

   Scope sOP = Scope::ByName("OP");
   CPPUNIT_ASSERT_EQUAL((size_t)8, sOP.MemberSize());
   CPPUNIT_ASSERT(sOP.FunctionMemberByName("operator MYINT"));
   Type tMYINT = Type::ByName("MYINT");
   CPPUNIT_ASSERT_EQUAL(tMYINT.Name(SCOPED), sOP.MemberByName("operator MYINT").TypeOf().ReturnType().Name(SCOPED));

   Scope sFUNCS = Scope::ByName("FUNCS");
   CPPUNIT_ASSERT(sFUNCS);
   CPPUNIT_ASSERT_EQUAL((size_t)2, sFUNCS.FunctionMemberSize());
   CPPUNIT_ASSERT(sFUNCS.FunctionMemberByName("operator-"));

   Type tFUNCSC = Type::ByName("FUNCS::C");
   CPPUNIT_ASSERT(tFUNCSC);
   CPPUNIT_ASSERT_EQUAL((size_t)9, tFUNCSC.FunctionMemberSize());

   Type tOpRet = sFUNCS.FunctionMemberByName("operator-").TypeOf().ReturnType();
   CPPUNIT_ASSERT(tOpRet);
   CPPUNIT_ASSERT_EQUAL(tFUNCSC.Name(SCOPED), tOpRet.Name(SCOPED));

   CPPUNIT_ASSERT(tFUNCSC.MemberByName("operator MYINT"));
   CPPUNIT_ASSERT_EQUAL(tMYINT.Name(SCOPED), tFUNCSC.MemberByName("operator MYINT").TypeOf().ReturnType().Name(SCOPED));

   CPPUNIT_ASSERT(tFUNCSC.MemberByName("operator OP::OPS"));
   Type tOP_OPS = sOP.SubTypeByName("OPS");
   CPPUNIT_ASSERT(tOP_OPS);
   
   CPPUNIT_ASSERT_EQUAL(tOP_OPS.Name(SCOPED), tFUNCSC.MemberByName("operator OP::OPS").TypeOf().ReturnType().Name(SCOPED));

} // testOperators


void
ReflexSimple2Test::testTypedefSelection() {
   Type t = Type::ByName("xmlTypedefSelection::TypedefXmlSelClass2");
   CPPUNIT_ASSERT(t);
   CPPUNIT_ASSERT(t.IsTypedef());

   Type t2 = t.ToType();
   CPPUNIT_ASSERT(t2);
   CPPUNIT_ASSERT(t2.IsTypedef());
   CPPUNIT_ASSERT_EQUAL(std::string("TypedefXmlSelClass"), t2.Name());

   Type t3 = t2.ToType();
   CPPUNIT_ASSERT(t3);
   CPPUNIT_ASSERT(t3.IsClass());
   CPPUNIT_ASSERT_EQUAL(std::string("RealXmlSelClass"), t3.Name());

   CPPUNIT_ASSERT_EQUAL(t3.Name(), t2.FinalType().Name());
   CPPUNIT_ASSERT_EQUAL(t3.Name(), t.FinalType().Name());

} // testTypedefSelection


void
ReflexSimple2Test::testTypedef() {
   Type t = Type::ByName("xmlTypedefSelection::TypedefXmlSelClass2");
   CPPUNIT_ASSERT(t);
   CPPUNIT_ASSERT_EQUAL(std::string("TypedefXmlSelClass"), t.ToType().Name());
   CPPUNIT_ASSERT_EQUAL(std::string("RealXmlSelClass"), t.ToType().ToType().Name());
   CPPUNIT_ASSERT_EQUAL(std::string("RealXmlSelClass"), t.FinalType().Name());
}


void
ReflexSimple2Test::testCppSelection() {
   Type t00 = Type::ByName("ns::TestTemplatedSelectionClass<int,int,float>");
   CPPUNIT_ASSERT(!t00);

   Type t01 = Type::ByName("ns::TestTemplatedSelectionClass<int,int>");
   CPPUNIT_ASSERT(t01);

   Type t02 = Type::ByName("ns::TestTemplatedSelectionClass<int,bool>");
   CPPUNIT_ASSERT(!t02);

   Scope g = Scope::ByName("");
   CPPUNIT_ASSERT(g);
   Scope s = Scope::ByName("ns");
   CPPUNIT_ASSERT(s);


   Member m0 = g.MemberByName("m_foo");
   CPPUNIT_ASSERT(m0);
   CPPUNIT_ASSERT(m0.IsDataMember());
   Type m0t = m0.TypeOf();
   CPPUNIT_ASSERT(m0t.IsFundamental());
   CPPUNIT_ASSERT_EQUAL(std::string("int"), m0t.Name());

   Member m1 = s.MemberByName("m_foo2");
   CPPUNIT_ASSERT(m1);
   CPPUNIT_ASSERT(m1.IsDataMember());
   Type m1t = m1.TypeOf();
   CPPUNIT_ASSERT(m1t.IsFundamental());
   CPPUNIT_ASSERT_EQUAL(std::string("int"), m1t.Name());

   Type t0 = Type::ByName("XYZ");
   CPPUNIT_ASSERT(t0);
   CPPUNIT_ASSERT(t0.IsEnum());

   Type t1 = Type::ByName("ns::ABC");
   CPPUNIT_ASSERT(t1);
   CPPUNIT_ASSERT(t1.IsEnum());

   Type t2 = Type::ByName("int (int)");
   CPPUNIT_ASSERT(t2);
   CPPUNIT_ASSERT(t2.IsFunction());

   Member m2 = g.MemberByName("foosq");
   CPPUNIT_ASSERT(m2);
   CPPUNIT_ASSERT(m2.IsFunctionMember());
   Type m2t = m2.TypeOf();
   CPPUNIT_ASSERT(m2t);
   CPPUNIT_ASSERT(m2t.IsFunction());
   CPPUNIT_ASSERT(t2.IsEquivalentTo(m2t));

   Member m3 = s.MemberByName("fooadd");
   CPPUNIT_ASSERT(m3);
   CPPUNIT_ASSERT(m3.IsFunctionMember());
   Type m3t = m3.TypeOf();
   CPPUNIT_ASSERT(m3t);
   CPPUNIT_ASSERT(m3t.IsFunction());
   CPPUNIT_ASSERT(t2.IsEquivalentTo(m3t));

} // testCppSelection


void
ReflexSimple2Test::testCppSelectNoAutoselect() {
   Type t = Type::ByName("ns::NoSelfAutoSelection");
   CPPUNIT_ASSERT(!t);
   Type t2 = Type::ByName("ns::AutoSelectClass");
   CPPUNIT_ASSERT(t2);
   CPPUNIT_ASSERT(t2.IsClass());

}


void
ReflexSimple2Test::testTypedefInClass() {
   Type t0 = Type::ByName("testclasses::WithTypedefMember");
   CPPUNIT_ASSERT(t0);
   Member t0m0 = t0.DataMemberByName("m_i");
   CPPUNIT_ASSERT(t0m0);
   CPPUNIT_ASSERT(t0m0.TypeOf().IsFundamental());
   CPPUNIT_ASSERT_EQUAL(std::string("int"), t0m0.TypeOf().Name());
   Member t0m1 = t0.DataMemberByName("m_mi");
   CPPUNIT_ASSERT(t0m1);
   CPPUNIT_ASSERT(t0m1.TypeOf().IsTypedef());
   CPPUNIT_ASSERT_EQUAL(std::string("MyInt"), t0m1.TypeOf().Name());
   CPPUNIT_ASSERT(t0m1.TypeOf().FinalType().IsFundamental());
   CPPUNIT_ASSERT_EQUAL(std::string("int"), t0m1.TypeOf().FinalType().Name());
   Member t0m2 = t0.DataMemberByName("m_v");
   CPPUNIT_ASSERT(t0m2);
   CPPUNIT_ASSERT(t0m2.TypeOf().IsClass());
   CPPUNIT_ASSERT(t0m2.TypeOf().IsTemplateInstance());
   const Type& tt0m2 = t0m2.TypeOf();
   CPPUNIT_ASSERT_EQUAL(std::string("std::vector<int>"), tt0m2.Name(SCOPED));
   CPPUNIT_ASSERT_EQUAL(std::string("int"), tt0m2.TemplateArgument_Begin()->Name());
   CPPUNIT_ASSERT_EQUAL(tt0m2.TemplateArgument_Begin()->Name(), (tt0m2.TemplateArgument_REnd() - 1)->Name());
   CPPUNIT_ASSERT_EQUAL(tt0m2.TemplateArgument_RBegin()->Name(), (tt0m2.TemplateArgument_End() - 1)->Name());
   TypeTemplate tt0 = tt0m2.TemplateFamily();
   CPPUNIT_ASSERT(tt0);
   CPPUNIT_ASSERT_EQUAL(std::string("vector"), tt0.Name());
   Member t0m3 = t0.DataMemberByName("m_mv");
   CPPUNIT_ASSERT(t0m3);
   CPPUNIT_ASSERT(t0m3.TypeOf().IsTypedef());
   CPPUNIT_ASSERT_EQUAL(std::string("MyVector"), t0m3.TypeOf().Name());
   CPPUNIT_ASSERT_EQUAL(std::string("testclasses::MyVector"), t0m3.TypeOf().Name(SCOPED));
   CPPUNIT_ASSERT(t0m3.TypeOf().FinalType().IsClass());
   CPPUNIT_ASSERT(t0m3.TypeOf().FinalType().IsTemplateInstance());
   CPPUNIT_ASSERT_EQUAL(std::string("vector<int>"), t0m3.TypeOf().FinalType().Name());
   CPPUNIT_ASSERT_EQUAL(std::string("std::vector<int>"), t0m3.TypeOf().FinalType().Name(SCOPED));

   Type t1 = Type::ByName("testclasses::WithTypedefMemberT<std::vector<int> >");
   CPPUNIT_ASSERT(t1);
   Member t1m0 = t1.DataMemberByName("m_t");
   CPPUNIT_ASSERT(t1m0);
   // KNOWN TO FAIL WITH GCCXML 0.9!
   // CPPUNIT_ASSERT(t1m0.TypeOf().IsTypedef());
   // CPPUNIT_ASSERT_EQUAL(std::string("testclasses::MyVector"), t1m0.TypeOf().Name(SCOPED));
   CPPUNIT_ASSERT(t1m0.TypeOf().FinalType().IsClass());
   CPPUNIT_ASSERT_EQUAL(std::string("std::vector<int>"), t1m0.TypeOf().FinalType().Name(SCOPED));

   Type t2 = Type::ByName("testclasses::WithTypedefMemberT<int>");
   CPPUNIT_ASSERT(t2);
   Member t2m0 = t2.DataMemberByName("m_t");
   CPPUNIT_ASSERT(t2m0);
   // KNOWN TO FAIL WITH GCCXML 0.9!
   // CPPUNIT_ASSERT(t2m0.TypeOf().IsTypedef());
   // CPPUNIT_ASSERT_EQUAL(std::string("testclasses::MyInt"), t2m0.TypeOf().Name(SCOPED));
   CPPUNIT_ASSERT(t2m0.TypeOf().FinalType().IsFundamental());
   CPPUNIT_ASSERT_EQUAL(std::string("int"), t2m0.TypeOf().FinalType().Name(SCOPED));

} // testTypedefInClass


void
ReflexSimple2Test::testConstMembers() {
   Type t = Type::ByName("testclasses::ConstNonConstMembers");
   CPPUNIT_ASSERT(t);
   CPPUNIT_ASSERT(t.IsClass());
   CPPUNIT_ASSERT(t.IsPublic());
   CPPUNIT_ASSERT(!t.IsProtected());
   CPPUNIT_ASSERT(!t.IsPrivate());
   CPPUNIT_ASSERT(!t.IsAbstract());
   CPPUNIT_ASSERT(!t.BaseAt(t.BaseSize()));
   CPPUNIT_ASSERT((*t.DataMember_Begin()) == (*(t.DataMember_REnd() - 1)));
   CPPUNIT_ASSERT((*(t.DataMember_End() - 1)) == (*t.DataMember_RBegin()));
   CPPUNIT_ASSERT((*t.FunctionMember_Begin()) == (*(t.FunctionMember_REnd() - 1)));
   CPPUNIT_ASSERT((*(t.FunctionMember_End() - 1)) == (*t.FunctionMember_RBegin()));
   CPPUNIT_ASSERT((*t.Member_Begin()) == (*(t.Member_REnd() - 1)));
   CPPUNIT_ASSERT((*(t.Member_End() - 1)) == (*t.Member_RBegin()));
   Member m0 = t.FunctionMemberByName("foo", Type::ByName("int (int)"));
   CPPUNIT_ASSERT(m0);
   CPPUNIT_ASSERT(!m0.TypeOf().IsConst());
   Member m1 = t.FunctionMemberByName("foo", Type(Type::ByName("int (int)"), CONST));
   CPPUNIT_ASSERT(m1);
   CPPUNIT_ASSERT(m1.TypeOf().IsConst());

   Member m2 = t.DataMemberByName("m_i");
   CPPUNIT_ASSERT(m2);
   CPPUNIT_ASSERT(!m2.IsConst());
   CPPUNIT_ASSERT(!m2.TypeOf().IsConst());
   Member m3 = t.DataMemberByName("m_ci");
   CPPUNIT_ASSERT(m3);
   CPPUNIT_ASSERT(m3.IsConst());
   CPPUNIT_ASSERT(m3.TypeOf().IsConst());

   Member m4 = t.MemberByName("constfoo");
   CPPUNIT_ASSERT(m4);
   CPPUNIT_ASSERT(m4.IsConst());
   CPPUNIT_ASSERT(m4.TypeOf().IsConst());
   CPPUNIT_ASSERT(!m4.IsVolatile());
   CPPUNIT_ASSERT(!m4.TypeOf().IsVolatile());

   Member m5 = t.MemberByName("nonconstfoo");
   CPPUNIT_ASSERT(m5);
   CPPUNIT_ASSERT(!m5.IsConst());
   CPPUNIT_ASSERT(!m5.TypeOf().IsConst());
   CPPUNIT_ASSERT(!m5.IsVolatile());
   CPPUNIT_ASSERT(!m5.TypeOf().IsVolatile());

   Member m6 = t.MemberByName("m_vi");
   CPPUNIT_ASSERT(m6);
   CPPUNIT_ASSERT(!m6.IsConst());
   CPPUNIT_ASSERT(!m6.TypeOf().IsConst());
   CPPUNIT_ASSERT(m6.IsVolatile());
   CPPUNIT_ASSERT(m6.TypeOf().IsVolatile());

} // testConstMembers


void
ReflexSimple2Test::testSubTypes() {
   Type t = Type::ByName("testclasses::WithTypedef");
   CPPUNIT_ASSERT(t);
   CPPUNIT_ASSERT(t.IsClass());

   CPPUNIT_ASSERT_EQUAL(size_t(1), t.SubTypeSize());

   Type st = t.SubTypeAt(0);
   CPPUNIT_ASSERT(st.IsTypedef());
   CPPUNIT_ASSERT_EQUAL(std::string("MyInt"), st.Name());
   CPPUNIT_ASSERT_EQUAL(std::string("testclasses::WithTypedef::MyInt"), st.Name(SCOPED));
   CPPUNIT_ASSERT_EQUAL(std::string("int"), st.ToType().Name());

   t = Type::ByName("std::vector<int>");
   CPPUNIT_ASSERT(t);
   CPPUNIT_ASSERT(t.IsClass());

   int tdefs = 0;
   CPPUNIT_ASSERT(t.SubTypeSize() > 0);

   for (Reverse_Type_Iterator ti = t.SubType_RBegin(); ti != t.SubType_REnd(); ++ti) {
      if (ti->IsTypedef()) {
         ++tdefs;
      }
   }

   CPPUNIT_ASSERT(5 < tdefs && tdefs < 20);

   Scope s0 = Scope::ByName("std::vector<int>");
   CPPUNIT_ASSERT(s0);
   CPPUNIT_ASSERT(s0.SubTypeByName("iterator"));
   CPPUNIT_ASSERT(s0.SubTypeByName("value_type"));

   Scope s1 = Scope::ByName("std::vector<MyClass>");
   CPPUNIT_ASSERT(s1);
   CPPUNIT_ASSERT(s1.SubTypeByName("iterator"));
   CPPUNIT_ASSERT(Type::ByName(s0.Name(SCOPED) + "::value_type"));
   CPPUNIT_ASSERT(s1.SubTypeByName("value_type"));

} // testSubTypes


void
ReflexSimple2Test::testToTypeFinal() {
   Type t = Type::ByName("testclasses::Typedefs");
   CPPUNIT_ASSERT(t);

   Type t0;

   for (Type_Iterator ti = t.SubType_Begin(); ti != t.SubType_End(); ++ti) {
      if ((*ti).Name() == "RPMYINT") {
         t0 = *ti;
      }
   }
   CPPUNIT_ASSERT(t0);
   CPPUNIT_ASSERT_EQUAL(std::string("RPMYINT"), t0.Name());
   CPPUNIT_ASSERT_EQUAL(std::string("testclasses::Typedefs::RPMYINT"), t0.Name(S | Q));
   CPPUNIT_ASSERT_EQUAL(std::string("int*"), t0.FinalType().Name());
   CPPUNIT_ASSERT_EQUAL(std::string("int* const&"), t0.FinalType().Name(S | Q));

   Type t1;

   for (Type_Iterator ti = t.SubType_Begin(); ti != t.SubType_End(); ++ti) {
      if ((*ti).Name() == "PPMYINT") {
         t1 = *ti;
      }
   }
   CPPUNIT_ASSERT(t1);
   CPPUNIT_ASSERT_EQUAL(std::string("PPMYINT"), t1.Name());
   CPPUNIT_ASSERT_EQUAL(std::string("testclasses::Typedefs::PPMYINT"), t1.Name(S | Q));
   CPPUNIT_ASSERT_EQUAL(std::string("int**"), t1.FinalType().Name());
   CPPUNIT_ASSERT_EQUAL(std::string("const int**"), t1.FinalType().Name(S | Q));

   Type t2;

   for (Type_Iterator ti = t.SubType_Begin(); ti != t.SubType_End(); ++ti) {
      if ((*ti).Name() == "PPPMYINT") {
         t2 = *ti;
      }
   }
   CPPUNIT_ASSERT(t2);
   CPPUNIT_ASSERT_EQUAL(std::string("PPPMYINT"), t2.Name());
   CPPUNIT_ASSERT_EQUAL(std::string("testclasses::Typedefs::PPPMYINT"), t2.Name(S | Q));
   CPPUNIT_ASSERT_EQUAL(std::string("const int** const*"), t2.FinalType().Name(Q));
   CPPUNIT_ASSERT_EQUAL(std::string("const int** const*"), t2.FinalType().Name(S | Q));

} // testToTypeFinal


void
ReflexSimple2Test::testScopeSubFuns() {
   Scope s = Scope::ByName("testclasses");
   CPPUNIT_ASSERT(s);

   Type t = s.SubTypeByName("Outer");
   CPPUNIT_ASSERT(t);
   CPPUNIT_ASSERT_EQUAL(std::string("testclasses::Outer"), t.Name(S));
   Type t1 = s.SubTypeByName("Outer::Inner");
   CPPUNIT_ASSERT(t1);
   CPPUNIT_ASSERT_EQUAL(std::string("testclasses::Outer::Inner"), t1.Name(S));
   CPPUNIT_ASSERT_EQUAL(size_t(1), t1.SubScopeSize());
   CPPUNIT_ASSERT((Type) t1 == (Type) t.SubScopeAt(0));
   CPPUNIT_ASSERT_EQUAL(t.SubScope_Begin()->Name(), (t.SubScope_REnd() - 1)->Name());
   CPPUNIT_ASSERT_EQUAL(t.SubScope_RBegin()->Name(), (t.SubScope_End() - 1)->Name());


   TypeTemplate tt = s.SubTypeTemplateByName("WithTypedefMemberT");
   CPPUNIT_ASSERT(tt);
   CPPUNIT_ASSERT_EQUAL(std::string("testclasses::WithTypedefMemberT"), tt.Name(S));
   CPPUNIT_ASSERT_EQUAL(size_t(2), tt.TemplateInstanceSize());

   Scope s0 = s.SubScopeByName("TemplFun");
   CPPUNIT_ASSERT(s0);
   CPPUNIT_ASSERT_EQUAL(std::string("TemplFun"), s0.Name());
   s0 = s.SubScopeByName("Outer::Inner");
   CPPUNIT_ASSERT(s0);
   CPPUNIT_ASSERT_EQUAL(std::string("testclasses::Outer::Inner"), s0.Name(S));


   //for (MemberTemplate_Iterator mti = s0.MemberTemplate_Begin(); mti != s0.MemberTemplate_End(); ++mti ) {
   //   std::cout << (*mti).Name(S|Q) << std::endl;
   //}

   // FIXME: gccxml 060_patch3 does not produce a demangled name of a symbol, while later versions do
   // this will allow to check whether a function is templated or not and produce member templates
   //MemberTemplate mt = s0.MemberTemplateByName("foo");
   //CPPUNIT_ASSERT(mt);
} // testScopeSubFuns


void
ReflexSimple2Test::testEnums() {
   Scope s = Type::ByName("Bla::Base");
   CPPUNIT_ASSERT(2);

   Type t1 = s.SubTypeByName("protectedEnum");
   CPPUNIT_ASSERT(t1);
   CPPUNIT_ASSERT(!t1.IsPublic());
   CPPUNIT_ASSERT(t1.IsProtected());
   CPPUNIT_ASSERT(!t1.IsPrivate());
   CPPUNIT_ASSERT_EQUAL(std::string("PA"), t1.DataMember_Begin()->Name());
   CPPUNIT_ASSERT_EQUAL(std::string("PC"), t1.DataMemberAt(t1.DataMemberSize() - 1).Name());
   CPPUNIT_ASSERT_EQUAL(std::string("PA"), t1.Member_Begin()->Name());
   CPPUNIT_ASSERT_EQUAL(t1.DataMember_Begin()->Name(), (t1.DataMember_REnd() - 1)->Name());
   CPPUNIT_ASSERT_EQUAL(t1.DataMember_RBegin()->Name(), (t1.DataMember_End() - 1)->Name());
   CPPUNIT_ASSERT_EQUAL(t1.Member_Begin()->Name(), (t1.Member_REnd() - 1)->Name());
   CPPUNIT_ASSERT_EQUAL(t1.Member_RBegin()->Name(), (t1.Member_End() - 1)->Name());
   Member m0 = t1.DataMemberByName("PB");
   CPPUNIT_ASSERT(m0);
   Member m1 = t1.MemberByName("PB");
   CPPUNIT_ASSERT(m1);
   CPPUNIT_ASSERT(m0 == m1);
   CPPUNIT_ASSERT(s == t1.DeclaringScope());
   CPPUNIT_ASSERT(t1.Properties());


   Type t2 = s.SubTypeByName("privateEnum");
   CPPUNIT_ASSERT(t2);
   CPPUNIT_ASSERT(!t2.IsPublic());
   CPPUNIT_ASSERT(!t2.IsProtected());
   CPPUNIT_ASSERT(t2.IsPrivate());

} // testEnums


void
ReflexSimple2Test::fundamentalType() {
   Type t0 = Type::ByName("int");
   CPPUNIT_ASSERT(t0);
   CPPUNIT_ASSERT_EQUAL(kINT, Tools::FundamentalType(t0));

   Type t1 = Type::ByName("unsigned int");
   CPPUNIT_ASSERT(t1);
   CPPUNIT_ASSERT_EQUAL(kUNSIGNED_INT, Tools::FundamentalType(t1));

   Type t2 = Type::ByName("unsigned long int");
   CPPUNIT_ASSERT(t2);
   CPPUNIT_ASSERT_EQUAL(kUNSIGNED_LONG_INT, Tools::FundamentalType(t2));

   Type t3 = Type::ByName("long unsigned int");
   CPPUNIT_ASSERT(t3);
   CPPUNIT_ASSERT_EQUAL(kUNSIGNED_LONG_INT, Tools::FundamentalType(t3));

   Type t4 = Type::ByName("signed int");
   CPPUNIT_ASSERT(t4);
   CPPUNIT_ASSERT_EQUAL(kINT, Tools::FundamentalType(t4));

   Type t5 = Type::ByName("long int");
   CPPUNIT_ASSERT(t5);
   CPPUNIT_ASSERT_EQUAL(kLONG_INT, Tools::FundamentalType(t5));

   Type t6 = Type::ByName("Bla::Base");
   CPPUNIT_ASSERT(t6);
   CPPUNIT_ASSERT_EQUAL(kNOTFUNDAMENTAL, Tools::FundamentalType(t6));

} // fundamentalType


void
ReflexSimple2Test::testArray() {
   //int a[5];

   Type t0 = Type::ByName("int[5][4][3][2][1]");
   CPPUNIT_ASSERT(t0);

   //Type t1 = Type::ByTypeInfo(typeid(a));
   //CPPUNIT_ASSERT(t1);

   //Type t2 = Type::ByTypeInfo(typeid(int[5]));
   //CPPUNIT_ASSERT(t2);

   Type t = Type::ByName("testclasses::WithArray");
   CPPUNIT_ASSERT(t);

   Object o = t.Construct();
   CPPUNIT_ASSERT(o);

   Member m = t.DataMemberByName("m_a");
   CPPUNIT_ASSERT(m);

   Type memType = m.TypeOf();
   CPPUNIT_ASSERT(memType);

   Type arrType = memType.ToType();
   CPPUNIT_ASSERT(arrType);

   void* mem = (char*) o.Address() + m.Offset();

   for (size_t i = 0; i < m.TypeOf().ArrayLength(); ++i) {
      CPPUNIT_ASSERT_EQUAL((int) i + 1, Object_Cast<int>(Object(arrType, mem)));
      mem = (char*) mem + arrType.SizeOf();

   }

   o.Destruct();
} // testArray


void
ReflexSimple2Test::testCommentsEtc() {
   Type t = Type::ByName("testclasses::WithTransientMember");
   CPPUNIT_ASSERT(t);

   Member m0 = t.MemberByName("m_transient");
   CPPUNIT_ASSERT(m0);
   CPPUNIT_ASSERT(m0.IsTransient());

   Member m1 = t.MemberByName("m_nottransient");
   CPPUNIT_ASSERT(m1);
   CPPUNIT_ASSERT(!m1.IsTransient());

}


void
ReflexSimple2Test::unloadType() {
   Type t0 = Type::ByName("ClassH");
   t0.UpdateMembers();
   CPPUNIT_ASSERT(t0);
   t0.Unload();
   CPPUNIT_ASSERT(!t0);
   CPPUNIT_ASSERT_EQUAL(std::string("ClassH"), t0.Name());

   Type t1 = Type::ByName("ClassC");
   CPPUNIT_ASSERT(t1);
   Member t1m0 = t1.MemberByName("c");
   CPPUNIT_ASSERT(t1m0);

}


void
ReflexSimple2Test::iterateVector() {
   std::vector<int> v;
   v.push_back(1);
   v.push_back(2);
   v.push_back(3);
   v.push_back(4);
   v.push_back(5);
   v.push_back(6);


   Type t = Type::ByName("std::vector<int>");
   CPPUNIT_ASSERT(t);
   Object o = Object(t, &v);

   size_t vsize = 0;
   o.Invoke("size", vsize);
   Type templParType0 = t.TemplateArgumentAt(0);
   CPPUNIT_ASSERT(templParType0);
   CPPUNIT_ASSERT_EQUAL(std::string("int"), templParType0.Name());

   std::vector<void*> params;
   Type polParType;

   for (size_t i = 0; i < vsize; ++i) {
      params.clear();
      params.push_back(&i);
      int ret = -42;
      int* pret = &ret;
      CPPUNIT_ASSERT_EQUAL(ret, *pret);
      // not very standard since int* != int&, but we have to do
      // this since 'There shall be no references to references'
      // —ISO/IEC 14882:1998(E), the ISO C++ standard, in section 8.3.2 [dcl.ref]
      o.Invoke("at", pret, params);
      CPPUNIT_ASSERT_EQUAL(v[i], *pret);
   }

} // iterateVector


void
ReflexSimple2Test::testClasses() {
   const Type& t0 = Type::ByName("testclasses::MyClass");
   CPPUNIT_ASSERT(t0);
   CPPUNIT_ASSERT(!t0.IsStruct());
   CPPUNIT_ASSERT(t0.IsClass());

   const Type& t1 = Type::ByName("testclasses::MyStruct");
   CPPUNIT_ASSERT(t1);
   CPPUNIT_ASSERT(t1.IsClass());
   CPPUNIT_ASSERT(t1.IsStruct());

}


void
ReflexSimple2Test::testTemplateTypedefs() {
   const Type& t0 = Type::ByName("testclasses2::WithTypedefMemberT<testclasses::MyInt>");
   CPPUNIT_ASSERT(t0);
   CPPUNIT_ASSERT(t0.IsTypedef());

   const Type& t01 = t0.ToType();
   CPPUNIT_ASSERT(t01);
   CPPUNIT_ASSERT(t01.IsClass());
   CPPUNIT_ASSERT(t01 == t0.FinalType());
   CPPUNIT_ASSERT_EQUAL(std::string("WithTypedefMemberT<int>"), t01.Name());

}


void
ReflexSimple2Test::unloadLibrary() {
#if defined(_WIN32)
   int ret = FreeLibrary(s_libInstance);

   if (ret == 0) {
      std::cout << "Unload of dictionary library failed. Reason: " << GetLastError() << std::endl;
   }
   CPPUNIT_ASSERT(ret);
#else
   int ret = dlclose(s_libInstance);

   if (ret == -1) {
      std::cout << "Unload of dictionary library failed. Reason: " << dlerror() << std::endl;
   }
   CPPUNIT_ASSERT(!ret);
#endif

   Type t = Type::ByName("ClassH");
   //CPPUNIT_ASSERT(!t);

   //std::cout << "Endless" << std::endl;
   //while (true) {}

} // unloadLibrary


// Class registration on cppunit framework
CPPUNIT_TEST_SUITE_REGISTRATION(ReflexSimple2Test);

// CppUnit test-driver common for all the cppunit test classes
#include <CppUnit_testdriver.cpp>
