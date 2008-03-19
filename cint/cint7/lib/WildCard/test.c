/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (cint@pcroot.cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
#include <stdio.h>

extern Tcl_Interp *interp;

main() {
  test();
  test1();
}

test() {
  int code;
  int i=123;
  double d=3.14;

  // Link C/C++ variable and Tcl variable
  Tcl_LinkVar(interp,"i",(char*)(&i),TCL_LINK_INT);
  Tcl_LinkVar(interp,"d",(char*)(&d),TCL_LINK_DOUBLE);

  printf("i=%s\n",Tcl_GetVar(interp,"i",0));
  printf("d=%s\n",Tcl_GetVar(interp,"d",0));

  Tcl_SetVar(interp,"i","456",0);
  Tcl_SetVar(interp,"d","1.41421356",0);
  printf("i=%d\n",i);
  printf("d=%g\n",d);

  i=3229;
  d=1.6e-19;
  code=Tcl_Eval(interp,"expr $i");
  if(*interp->result!=0) printf("%s\n",interp->result);
  code=Tcl_Eval(interp,"expr $d");
  if(*interp->result!=0) printf("%s\n",interp->result);

  if(code!=TCL_OK) exit(1);
}

test1()
{
  int i2=123;
  double d2=3.14;
#pragma tcl interp i2 d2
  puts $i2
  puts $d2
  set i2 456
  set d2 6.24
#pragma endtcl
  printf("d2=%g i2=%d\n",d2,i2); 
}
