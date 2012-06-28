/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (root-cint@cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
// UserMain.cxx

#include <stdio.h>
#include "G__ci.h"
#include "UserMain.h"   /* host.h can be an empty file */

//////////////////////////////////////////////////////////////////////
// main application 
//////////////////////////////////////////////////////////////////////
void TheSimplestExample() {
  G__init_cint("UserMain script.cxx");
  G__scratch_all();
}

void LoadScriptfileAfterwards() {
  int state;
  state=G__init_cint("cint");
  switch(state) {
  case G__INIT_CINT_SUCCESS_MAIN:
    /* Should never happen */
    break;
  case G__INIT_CINT_SUCCESS:
    state=G__loadfile("script.cxx");
    if(state==G__LOADFILE_SUCCESS) {
      // G__calc and G__exec_text API can be used to evaluate C/C++ command
      // Read doc/ref.txt for those APIs.
      G__calc("script(\"Calling from compiled main application 1\")");
      G__exec_text("script(\"Calling from compiled main application 2\")");
    }
    break;
  case G__INIT_CINT_FAILURE:
  default:
    printf("Cint initialization failed.\n");
  }
  G__scratch_all();
}

int main() {
  TheSimplestExample();
  LoadScriptfileAfterwards();
}

//////////////////////////////////////////////////////////////////////
// precompiled library
//////////////////////////////////////////////////////////////////////
void f1(int i) {
  printf("f1(%d)\n",i);
}

void f2(double d) {
  printf("f2(%g)\n",d);
}

