#ifdef __CLING__

#pragma link off all globals;
#pragma link off all classes;
#pragma link off all functions;

#pragma link C++ nestedclass;

#pragma link C++ namespace Name;

#pragma link C++ class Name::MyClass;
#pragma link C++ class TEST_N::X<float, 2>+;

#endif
