/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (root-cint@cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
// mtlib.h  : Multi thread demo program on Windows-NT/9x Visual C++
//
//  Usage:
//   makecint -mk makethread -dl thread.dll -H mtlib.h
//   make -f makethread
//   cint mtmain.cxx

#ifndef __CINT__
#include <iostream.h>
#endif
#include <windows.h>

// Data for running thread function /////////////////////////////////
class FuncData {
 public:
  FuncData(int nin,char *msgin,LPCRITICAL_SECTION lpcin) 
    { n=nin; msg=msgin; lpc=lpcin; }
  int n;
  char *msg;
  LPCRITICAL_SECTION lpc;
};

// Precompiled thread function //////////////////////////////////////
inline int PrecompiledFunc(void *p) {
  int i;
  FuncData *px=(FuncData*)p;
  cout << "ThreadID=" << GetCurrentThreadId() << endl;
  for(i=0;i<px->n;i++) {
    EnterCriticalSection(px->lpc);
    cout << px->msg << " i=" << i << endl;    
    LeaveCriticalSection(px->lpc);
  }
  return(i);
}
