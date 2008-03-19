/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (cint@pcroot.cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
// compiled.cxx , precompiled class/function definition
#include "compiled.h"

///////////////////////////////////////////////////////////////////////
// Compiled1 member function
///////////////////////////////////////////////////////////////////////
Compiled1::Compiled1() {
  publicData1 = 123;
  protectedData1 = 456;
  privateData1 = 789;
}

void Compiled1::publicFunc1() { 
  printf("Compiled1::publicFunc1()\n");
}

void Compiled1::protectedFunc1() { 
  printf("Compiled1::protectedFunc1()\n");
}

void Compiled1::privateFunc1() { 
  printf("Compiled1::privateFunc1()\n");
}

///////////////////////////////////////////////////////////////////////
// Compiled Global functions
///////////////////////////////////////////////////////////////////////
void compiledVirtualFuncCall(Compiled1** obj,int n) { 
  printf("--- Compiled Virtual Function Call ---\n");
  for(int i=0;i<n;i++) {
    // Virtual function call for Compiled1 
    obj[i]->publicFunc1(); 
  }
}

void callInterpretedFuncFromCompiledFunc() { // compiled
  // interpretedGlobalFunc1() called from compiled code
  extern void interpretedGlobalStubFunc1();
  interpretedGlobalStubFunc1();
}


