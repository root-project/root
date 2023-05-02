#ifndef MYCLASS1_H
#define MYCLASS1_H
// myclass.h 
#include "TNamed.h" 
class myclass: public TNamed 
{ 
 public: 
  Double32_t a; 
  myclass(); 
  myclass(const char* name); 
  ClassDefOverride(myclass,1); 
}; 

#ifdef __MAKECINT__ 
#pragma link C++ class myclass+; 
#endif

#endif
