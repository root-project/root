/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (root-cint@cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
// compiled.h ,  precompiled class header file
#ifndef COMPILED_H
#define COMPILED_H

#include <stdio.h>

class Compiled1 {
 public:
  Compiled1();
  virtual void publicFunc1();
  int publicData1;
 protected:
  void protectedFunc1();
  int protectedData1;
 private:
  void privateFunc1();
  int privateData1;
};

void compiledVirtualFuncCall(Compiled1** obj,int n) ; 
void callInterpretedFuncFromCompiledFunc() ;

// Need following pragma to allow protected member access from the interpreter
#ifdef __MAKECINT__
#pragma link C++ class+protected Compiled1;
#endif


#endif
