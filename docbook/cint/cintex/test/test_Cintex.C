#include <stdio.h>
#ifndef __CINT__
#include "dict/CintexTest.h"
#include "TSystem.h"
#include "Cintex/Cintex.h"
#endif

template <class T1, class T2> 
void failUnlessEqual( T1 r, T2 e, const char* c = "") { 
  if( r != e ) {
  cout << "Test failed in " << c << " : got " << r << " expected " << e << endl;
  assert(false);
  }
}
template <class T1> 
void failUnless( T1 r, const char* c = "") { 
  if( ! r ) {
  cout << "Test failed in " << c << " : got " << r << endl;
  assert(false);
  }
}

bool test_DoSomething() {
  A::B::C::MyClass object;
  failUnlessEqual(object.magic(),987654321,"DoSomething: magic value");
  int result = object.doSomething("Hello World");
  failUnlessEqual( result, 11, "DoSomething: incorrect return value from doSomething");
  return true;
}

bool test_PrimitiveArgs() {
  A::B::C::Primitives p;
  p.set_b(false);
  failUnlessEqual( p.b(), false, "PrimitiveArgs: fail to set bool to false");
  p.set_b(true);
  failUnlessEqual( p.b(), true, "PrimitiveArgs: fail to set bool to true");
  p.set_b(true);
  failUnlessEqual( p.b(), true, "PrimitiveArgs: fail to set bool to true");
  p.set_b(10);
  failUnlessEqual( p.b(), true, "PrimitiveArgs: fail to set bool to 10");
  p.set_b(0);
  failUnlessEqual( p.b(), false, "PrimitiveArgs: fail to set bool to 0");
//    self.failUnlessRaises( TypeError, p.set_b, 10 )
  p.set_c('h');
  failUnlessEqual( p.c(), 'h', "PrimitiveArgs: fail to set char");
  p.set_c(40);
  failUnlessEqual( p.c(), 40, "PrimitiveArgs: fail to set char");
//    self.failUnlessRaises( TypeError, p.set_c, 'ssss')    
  p.set_s(-8);
  failUnlessEqual( p.s(),-8, "PrimitiveArgs: fail to set short" );
//    self.failUnlessRaises( TypeError, p.set_s, 1.5)    
//    self.failUnlessRaises( TypeError, p.set_s, 'ssss')    
  p.set_i(8);
  failUnlessEqual( p.i(), 8, "PrimitiveArgs: fail to set int" );
  p.set_i(8.0);
  failUnlessEqual( p.i(), 8, "PrimitiveArgs: fail to set int from double" );
  p.set_l(-8);
  failUnlessEqual( p.l(),-8, "PrimitiveArgs: fail to set long" );
  p.set_uc(8);
  failUnlessEqual( p.uc(), 8, "PrimitiveArgs: fail to set unsigned char" );
  p.set_us(8);
  failUnlessEqual( p.us(), 8, "PrimitiveArgs: fail to set unsigned short" );
  p.set_ui(8);
  failUnlessEqual( p.ui(), 8, "PrimitiveArgs: fail to set unsigned int" );
  p.set_ul(8);
  failUnlessEqual( p.ul(), 8, "PrimitiveArgs: fail to set unsigned long" );
  p.set_f(8.8F);
  failUnlessEqual( p.f(), 8.8F, "PrimitiveArgs: fail to set float" );
  p.set_d(8.8);
  failUnlessEqual( p.d(), 8.8, "PrimitiveArgs: fail to set double" );
  p.set_d(8);
  failUnlessEqual( p.d(), 8., "PrimitiveArgs: fail to set double from int" );
  p.set_str("This is a string");
  failUnlessEqual( p.str(), "This is a string", "PrimitiveArgs: fail to set string" );
  p.set_cstr("This is a C string");
  failUnless( strcmp(p.cstr(), "This is a C string") == 0, "PrimitiveArgs: fail to set string" );
  failUnless( strcmp(p.ccstr(),"This is a C string") == 0, "PrimitiveArgs: fail to set string" );
  p.set_all(1, 'g', 7, 7, 7, 7.7F, 7.7, "Another string");
  failUnlessEqual( p.b(), 1);
  failUnlessEqual( p.c(), 'g');
  failUnlessEqual( p.s(), 7);
  failUnlessEqual( p.i(), 7);
  failUnlessEqual( p.l(), 7);
  failUnlessEqual( p.str(), "Another string");
  return true;
}

bool test_ReturnModes() {
  A::B::C::MyClass myobj;
  myobj.setMagic(1234567890);
  failUnlessEqual( myobj.magic(), 1234567890, "ReturnModes: creating object" );
  A::B::C::Calling  calling;
  calling.setByValue(myobj);
  failUnlessEqual( calling.retByValue().magic(), 1234567890 , "ReturnModes: fail return by value");
  failUnlessEqual( calling.retByPointer()->magic(), 1234567890 , "ReturnModes: fail return by pointer");
  failUnlessEqual( calling.retByReference().magic(), 1234567890 , "ReturnModes: fail return by reference");
  //failUnlessEqual( calling.retByRefPointer()->magic(), 1234567890 , "ReturnModes: fail return by reference pointer");
  return true;
}

bool test_UnknownTypes() {
  A::B::C::Calling calling;
  //---Returning unknown types
  void* rp = calling.retUnkownTypePointer();
  failUnless( rp );
  void* rr = (void*)&(calling.retUnkownTypeReference());
  failUnless( rr );
  //---Passing unknown types
  failUnlessEqual( calling.setByUnknownTypePointer(rp), 0x12345678);
  //failUnlessEqual( calling.setByUnknownTypeReference(rr), 0x12345678);
  //failUnlessEqual( calling.setByUnknownConstTypePointer(rp), 0x12345678);
  //failUnlessEqual( calling.setByUnknownConstTypeReference(rr), 0x12345678);
  return true;
}

bool test_CallingModes() {
  A::B::C::MyClass myobj;
  A::B::C::Calling calling;
  myobj.setMagic(22222222);
  //---Check calling modes-------------
  calling.setByValue(myobj);
  failUnlessEqual( calling.retByPointer()->magic(), 22222222 , "CallingModes: fail set by value");
  myobj.setMagic(33333333);
  calling.setByPointer(&myobj);
  failUnlessEqual( calling.retByPointer()->magic(), 33333333 , "CallingModes: fail set by pointer");
  failUnlessEqual( myobj.magic(), 999999 , "CallingModes: fail set by pointer");
  calling.setByPointer(0);
  failUnlessEqual( calling.retByPointer()->magic(), 0 , "CallingModes: fail set by null pointer");
  myobj.setMagic(44444444);
  calling.setByReference(myobj);
  failUnlessEqual( calling.retByPointer()->magic(), 44444444 , "CallingModes: fail set by reference");
  failUnlessEqual( myobj.magic(),  999999 , "CallingModes: fail set by reference");
  myobj.setMagic(44445555);
  calling.setByConstReference(myobj);
  failUnlessEqual( calling.retByPointer()->magic(), 44445555 , "CallingModes: fail set by reference");
  failUnlessEqual( myobj.magic(),  44445555 , "CallingModes: fail set by reference");
  myobj.setMagic(55555555);
  A::B::C::MyClass *p = &myobj;
  calling.setByRefPointer(p);
  failUnlessEqual( calling.retByPointer()->magic(), 55555555 , "CallingModes: fail set by reference pointer");
  failUnlessEqual( myobj.magic(), 999999 , "CallingModes: fail set by reference pointer");
  failUnlessEqual( calling.retStrByValue(), "value" );
  failUnlessEqual( calling.retStrByRef(), "reference" );
  failUnlessEqual( calling.retStrByConstRef(), "const reference" );
  failUnless( strcmp(calling.retConstCStr(), "const pointer") == 0, "CallingModes: fail return C string");
  failUnless( strcmp(calling.retCStr(), "pointer") == 0 );
  // del myobj, calling
  failUnlessEqual( A::B::C::MyClass::instances(), 2,  "CallingModes: MyClass instances not deleted");
  failUnlessEqual( A::B::C::s_public_instances, 2,  "CallingModes: MyClass s_public_instances not correctly read");
  return true;
}

bool test_DefaultArguments()  {
  A::B::C::DefaultArguments m0;  // default arguments (1, 0.0)
  failUnlessEqual(m0.i(), 1 );
  failUnlessEqual(m0.f(), 0.0);
  A::B::C::DefaultArguments m1(99);
  failUnlessEqual(m1.i(), 99);
  failUnlessEqual(m1.f(), 0.0);
  A::B::C::DefaultArguments m2(88, 8.8);
  failUnlessEqual(m2.i(), 88);
  failUnlessEqual(m2.f(), 8.8);
  failUnlessEqual( m1.function("string"), 1004.0);
  failUnlessEqual( m1.function("string",10.0), 1015.0);
  failUnlessEqual( m1.function("string",20.0,20), 46.0);
  failUnless( strcmp( m1.f_char(), "string value") == 0);
  failUnlessEqual( m1.f_string(), string("string value"));
  failUnlessEqual( m1.f_defarg().i(), 9 );
  failUnlessEqual( m1.f_defarg().f(), 9.9 );
//    self.assertRaises(TypeError, m2.function, (20.0,20))
//    self.assertRaises(TypeError, m2.function, ('a',20.0,20.5))
//    self.assertRaises(TypeError, m2.function, ('a',20.0,20,'b'))
  return true;
}

bool test_TypedefArguments()  {
  A::B::C::TypedefArguments m(99, 88.88);
  failUnlessEqual(m.i(), 99 );
  failUnlessEqual(m.f(), 88.88);
  MyInt myint;
  failUnlessEqual( m.function(myint, 20.0, 20), 40.0);
  return true;
}

bool test_Inheritance()  {
  A::B::C::Derived d0;
  //--- Data members
  d0.base1 = 1;
  d0.base2 = 2;
  d0.a = 10;
  d0.f = 99.;
  //Down cast
  A::B::C::Base1* b1 = &d0;
  A::B::C::Base2* b2 = &d0;
  failUnlessEqual(b1->base1, 1 );
  failUnlessEqual(b2->base2, 2 );
  //Up cast
  A::B::C::Derived* d1 = (A::B::C::Derived*)b1;
  failUnlessEqual(d1->a, 10 );
  failUnlessEqual(d1->f, 99. );
  // Non-virtual functions
  failUnlessEqual(d1->getBase(), 10 );
  failUnlessEqual(b1->getBase(), 1 );
  failUnlessEqual(b2->getBase(), 2 );
  // Virtual functions
  failUnlessEqual(d0.v_getBase(), 10 );
  failUnlessEqual(d0.v_get1(), 1 );
  failUnlessEqual(d0.v_get2(), 2 );
  failUnlessEqual(b1->v_getBase(), 10 );
  failUnlessEqual(b2->v_getBase(), 10 );
  return true;
}

bool test_MethodOverloading() {
  A::B::C::Calling calling;
  failUnlessEqual(calling.overloaded(10), 1);
  failUnlessEqual(calling.overloaded(10.0F), 2);
  failUnlessEqual(calling.overloaded(10, 10.0F), 3);
  failUnlessEqual(calling.overloaded(10.0F, 10), 4);
  return true;
}
namespace A{namespace B {namespace C{}}}

using namespace A::B::C;

bool test_MethodOperators() {
  failUnlessEqual(Number(20) + Number(10), Number(30) );
  failUnlessEqual(Number(20) - Number(10), Number(10) );
  failUnlessEqual(Number(20) / Number(10), Number(2) );
  failUnlessEqual(Number(20) * Number(10), Number(200) );
  failUnlessEqual(Number(5)  & Number(14), Number(4) );
  failUnlessEqual(Number(5)  | Number(14), Number(15) );
  failUnlessEqual(Number(5)  ^ Number(14), Number(11) );
  failUnlessEqual(Number(5)  << 2, Number(20) );
  failUnlessEqual(Number(20) >> 2, Number(5) );
  Number  n  = Number(20);
  n += Number(10);
  n -= Number(10);
  n *= Number(10);
  n /= Number(2);
  failUnlessEqual(n ,Number(100) );
  failUnlessEqual(Number(20) >  Number(10), 1 );
  failUnlessEqual(Number(20) <  Number(10), 0 );
  failUnlessEqual(Number(20) >= Number(20), 1 );
  failUnlessEqual(Number(20) <= Number(10), 0 );
  failUnlessEqual(Number(20) != Number(10), 1 );
  failUnlessEqual(Number(20) == Number(10), 0 );
  return true;
}

bool test_TemplatedClasses() {
  A::B::C::Template<A::B::C::MyClass,float> o1;
  GlobalTemplate<A::B::C::MyClass,float> o2;
  A::B::C::Template<A::B::C::MyClass,unsigned long> o3;
  int r1 = o1.doSomething("Hello World");
  failUnlessEqual( r1, strlen("Hello World"));
  int r2 = o2.doSomething("Hello World");
  failUnlessEqual( r2, strlen("Hello World"));
  int r3 = o3.doSomething("Hello World");
  failUnlessEqual( r3, strlen("Hello World"));
  return true;
}


bool test_VirtualInheritance() {
  A::B::C::Diamond d;
  failUnlessEqual(d.vf(), 999.999 );
  failUnlessEqual(d.magic(), 987654321);
  return true;
}

bool test_IsA() {
  using namespace A::B::C;
  Diamond real;
  Diamond* d = &real;
  Virtual* v = &real;
 
  failUnless(strcmp(d->IsA()->GetName(), "A::B::C::Diamond")==0);
  failUnless(strcmp(v->IsA()->GetName(), "A::B::C::Diamond")==0);
  
  IntermediateA  ireal;
  IntermediateA* ia = &ireal;
  Virtual*       va = &ireal;
  failUnless(strcmp(ia->IsA()->GetName(), "A::B::C::IntermediateA")==0);
  failUnless(strcmp(va->IsA()->GetName(), "A::B::C::IntermediateA")==0);
  
  return true;
}


bool test_ShowMembers() {
  return true;
}

bool test_NestedClasses() {
  A::B::C::MyClass d;
  A::B::C::Template<A::B::C::MyClass,float>::Nested obj;
  obj.first = d;
  failUnlessEqual(obj.first.magic(),987654321,"NestedClasses: magic value");
  return true;
}

int half(int i) { return i/2; }
int triple(int i) { return i*3; }

bool test_FunctionPointer() {
  A::B::C::Calling calling;
  //failUnlessEqual(calling.call(half,10), 5 );
  //failUnlessEqual(calling.call(triple,10), 30 );
  return true;
}

bool test_ObjectArrays() {
  int instances = A::B::C::MyClass::instances();
  failUnlessEqual( A::B::C::MyClass::instances(), instances,  "ObjectArrays: MyClass instances not zero");
  A::B::C::MyClass* myarray = new A::B::C::MyClass[10];
  failUnlessEqual( A::B::C::MyClass::instances(), instances+10,  "ObjectArrays: MyClass instances not 10");
  delete [] myarray;
  failUnlessEqual( A::B::C::MyClass::instances(), instances,  "ObjectArrays: MyClass instances not clean");
  return true;
}

bool test_Enums() {
  failUnlessEqual( MyNS::one, 1,  "Enums: value one");
  failUnlessEqual( MyNS::two, 2,  "Enums: value two");
  failUnlessEqual( one, 1,  "Enums: global value one");
  failUnlessEqual( two, 2,  "Enums: global value two");
  return true;
}

bool test_Variables() {
  failUnlessEqual( gMyInt, 123,  "Variables: global int");
  failUnlessEqual( A::gMyPointer, (void*)0,  "Variables: namespace pointer");
  return true;
}

using namespace ROOT::Cintex;

void test_Cintex()
{
  //gDebug = 2;
  //gSystem->Load("libReflex");
  //gSystem->Load("libCintex");
  Cintex::Enable();
  Cintex::SetDebug(0);
  //gSystem->Load("test_CintexTestRflx");
  gROOT->ProcessLine("#include <vector>");

  cout << "DoSomething: "        << (test_DoSomething()        ? "OK" : "FAIL") << endl;
  cout << "PrimitiveArgs: "      << (test_PrimitiveArgs()      ? "OK" : "FAIL") << endl;
  cout << "ReturnModes: "        << (test_ReturnModes()        ? "OK" : "FAIL") << endl;
  cout << "CallingModes: "       << (test_CallingModes()       ? "OK" : "FAIL") << endl;
  cout << "UnknownTypes: "       << (test_UnknownTypes()       ? "OK" : "FAIL") << endl;
  cout << "DefaultArguments: "   << (test_DefaultArguments()   ? "OK" : "FAIL") << endl;
  cout << "TypedefArguments: "   << (test_TypedefArguments()   ? "OK" : "FAIL") << endl;
  cout << "Inheritance: "        << (test_Inheritance()        ? "OK" : "FAIL") << endl;
  cout << "VirtualInheritance: " << (test_VirtualInheritance() ? "OK" : "FAIL") << endl;
  cout << "MethodOverloading: "  << (test_MethodOverloading()  ? "OK" : "FAIL") << endl;
  cout << "MethodOperators: "    << (test_MethodOperators()    ? "OK" : "FAIL") << endl;
  cout << "TemplatedClasses: "   << (test_TemplatedClasses()   ? "OK" : "FAIL") << endl;
  cout << "IsA: "                << (test_IsA()                ? "OK" : "FAIL") << endl;
  cout << "ShowMembers: "        << (test_ShowMembers()        ? "OK" : "FAIL") << endl;
  cout << "NestedClasses: "      << (test_NestedClasses()      ? "OK" : "FAIL") << endl;
  cout << "FunctionPointer: "    << (test_FunctionPointer()    ? "OK" : "FAIL") << endl;

  cout << "ObjectArrays: "       << (test_ObjectArrays()       ? "OK" : "FAIL") << endl;
  cout << "Enums: "              << (test_Enums()              ? "OK" : "FAIL") << endl;
  cout << "Variables: "          << (test_Variables()          ? "OK" : "FAIL") << endl;
}
