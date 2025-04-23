#ifdef __CINT__

#pragma link off all globals;
#pragma link off all classes;
#pragma link off all functions;

//Added
#pragma link C++ nestedclasses;

#pragma link C++ namespace A;
#pragma link C++ namespace B;

#pragma link C++ class A::Class1;
#pragma link C++ class A::Class2<A::Class1>!-;
#pragma link C++ class B::Class3+;

#endif
