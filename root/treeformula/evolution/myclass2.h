#ifndef MYCLASS2_H
#define MYCLASS2_H
// myclass.h 
#include "TNamed.h" 
class myclass: public TNamed 
{ 
 public: 
  Double_t a;
  Double_t GetA() { return a; }
  myclass(); 
  myclass(const char* name); 
  ClassDefOverride(myclass,2); 
}; 

#ifdef __MAKECINT__ 
#pragma link C++ class myclass+; 
#endif

#endif
