// myclass.cc 

#include "myclass1.h" 
ClassImp(myclass) 
myclass::myclass():a(0) {} 
myclass::myclass(const char* name):a(0){SetName(name);}
