/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (cint@pcroot.cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/

#include <stdio.h>

///////////////////////////////////////////////////////////////////
// a simple class to manage error dump file
class DumpFile {
  FILE *fp;
 public:
  DumpFile() {fp = fopen("dump.out","w");}
  ~DumpFile() { fclose(fp);}
  void Output(char *msg) {
    fprintf(fp,"%s",msg);
  }
} DumpFileObj;

///////////////////////////////////////////////////////////////////
// Error message receiver function that has to be given to CINT API
extern "C" void ErrorReceiver(char *msg) {
  DumpFileObj.Output(msg);
}

///////////////////////////////////////////////////////////////////
// Error reaction function that has to be given to CINT API
extern "C" void stopProcess() {
  printf("processed by the error callback\n");
  exit(-1);
}

///////////////////////////////////////////////////////////////////
#ifndef __CINT__
// Prototype for needed CINT API. Alternatively, you can include G__ci.h
// Refer to doc/ref.txt for detail of below API function
extern "C" void G__set_errmsgcallback(void *p2f);
extern "C" void G__set_aterror(void (*p2f)());
#endif

///////////////////////////////////////////////////////////////////
// Initialize routine to start error redirection
void InitializeDump() {
  G__set_errmsgcallback(ErrorReceiver);
  G__set_aterror(stopProcess);
}

