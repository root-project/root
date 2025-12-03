#ifdef __CINT__

#pragma link off all globals;
#pragma link off all classes;
#pragma link off all functions;

#pragma link C++ nestedclass;

#pragma link C++ namespace Master;

#pragma link C++ class Master::Container;
#pragma link C++ class Master::Object;

#pragma link C++ function Master::Container::func(Master::Object *);


#endif
