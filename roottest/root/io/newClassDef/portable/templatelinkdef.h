#ifdef __CINT__

#pragma link off all globals;
#pragma link off all classes;
#pragma link off all functions;
#pragma link C++ nestedclasses;
#pragma link C++ nestedtypedefs;

#pragma link C++ class MyTemplate<int>+;
#pragma link C++ class MyTemplate<const double*>+;
#pragma link C++ class MyPairTemplate<int, double>+;
#pragma link C++ class MyPairTemplate<int, int>+;

#pragma link C++ function template_driver;

#endif
