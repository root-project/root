#ifdef __CINT__

#pragma link off all globals;
#pragma link off all classes;
#pragma link off all functions;

#pragma link C++ nestedclasses;
#pragma link C++ namespace Slave1;
#pragma link C++ class Slave1::Object;

#pragma link C++ function Master::Container::func(Slave1::Object *);

#endif
