#ifdef __CINT__

#pragma link off all globals;
#pragma link off all classes;
#pragma link off all functions;

#pragma link C++ nestedclasses;

#pragma link C++ namespace Slave2;
#pragma link C++ class Slave2::Object;

#pragma link C++ function Master::Container::func(Slave2::Object *);

#endif
