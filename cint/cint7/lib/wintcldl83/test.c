/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (cint@pcroot.cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
#include <stdio.h>

main() {
  test();
}

test() {
  Tcl_Interp *interp;
  int code;
  int i=123;
  double d=3.14;

  interp = Tcl_CreateInterp();
  Tcl_AppInit(interp);
  // cinttk_init();

  // Link C/C++ variable and Tcl variable
  Tcl_LinkVar(interp,"i",(char*)(&i),TCL_LINK_INT);
  Tcl_LinkVar(interp,"d",(char*)(&d),TCL_LINK_DOUBLE);

  printf("i=%s\n",Tcl_GetVar(interp,"i",0));
  printf("d=%s\n",Tcl_GetVar(interp,"d",0));

  Tcl_SetVar(interp,"i","456",0);
  Tcl_SetVar(interp,"d","1.41421356",0);
  printf("i=%d\n",i);
  printf("d=%g\n",d);

  code=Tcl_Eval(interp,"set i 789");
  code=Tcl_Eval(interp,"set d 0.71");
  printf("i=%d\n",i);
  printf("d=%g\n",d);

  i=3229;
  d=1.6e-19;
  code=Tcl_Eval(interp,"expr $i");
  if(*interp->result!=0) printf("%s\n",interp->result);
  code=Tcl_Eval(interp,"expr $d");
  if(*interp->result!=0) printf("%s\n",interp->result);

  printf("tcl source code insertion test\n");

#pragma tcl interp
  set i 512
  set d 299.793
#pragma endtcl

  printf("i=%d\n",i);
  printf("d=%g\n",d);

  if(code!=TCL_OK) exit(1);
  exit(0);
}

