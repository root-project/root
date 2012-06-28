/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (root-cint@cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
#include "p2f.h"

/* example of interpreted function , slow */
int InterpretedFunc(char *s,double d) {
  int result;
  result = sprintf(s,"InterpretedFunc %g\n",d);
  return(result);
}

/* Example of bytecode compiled function , faster */
#pragma bytecode
int BytecodeFunc(char *s,double d) {
  int result;
  result = sprintf(s,"BytecodeFunc %g\n",d);
  return(result);
}
#pragma endbytecode

/* Example of native compiled function, very fast , 
* if your OS does not support sharedlib, bytecode version will be used */
#pragma bytecode
#pragma compile
int TryNativeThenBytecode(char *s,double d) {
  int result;
  result = sprintf(s,"TryNativeThenBytecode %g\n",d);
  return(result);
}
#pragma endcompile
#pragma endbytecode

main() {
  char string[100];
  int result;
  ExecP2F a;
  void *p;

  a.SetP2F((void*)CompiledFunc);
  result=a.DoP2F(string,1.23);
  printf("result=%d string=%s\n",result,string);

  a.SetP2F((void*)InterfaceMethod);
  result=a.DoP2F(string,4.56);
  printf("result=%d string=%s\n",result,string);

  a.SetP2F((void*)BytecodeFunc);
  result=a.DoP2F(string,7.89);
  printf("result=%d string=%s\n",result,string);

  a.SetP2F((void*)InterpretedFunc);
  result=a.DoP2F(string,0.12);
  printf("result=%d string=%s\n",result,string);

  a.SetP2F((void*)TryNativeThenBytecode);
  result=a.DoP2F(string,3.45);
  printf("result=%d string=%s\n",result,string);
}
