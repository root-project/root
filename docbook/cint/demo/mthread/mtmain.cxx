/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (cint@pcroot.cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
// mtmain.cxx : Multi thread demo program on Windows-NT/9x Visual C++
//
//  Usage:
//   makecint -mk makethread -dl thread.dll -H mtlib.h
//   make -f makethread
//   cint mtmain.cxx

#include <windows.h>
#include <iostream.h>
#include "mtlib.dll"

// Interpreted foreground function //////////////////////////////////////
int InterpretedFunc(void *p) {
  struct FuncData *px=(struct FuncData*)p;
  cout << "ThreadID=" << GetCurrentThreadId() << endl;
  for(int i=0;i<px->n;i++) {
    EnterCriticalSection(px->lpc);
    cout << px->msg << " i=" << i << endl;    
    LeaveCriticalSection(px->lpc);
  }
  return(i);
}

// Running 1 foreground and 1 background job ///////////////////////////
void test1() {
  CRITICAL_SECTION c;
  InitializeCriticalSection(&c);

  // create data for jobs 
  FuncData background(100,"Compiled backgrond",&c);
  FuncData interpreted(50,"Interpreted foreground",&c);
  FuncData foreground(50,"Compiled foreground",&c);

  // start background job
  DWORD ThreadID;
  HANDLE h = CreateThread(NULL,0
                         ,(LPTHREAD_START_ROUTINE)PrecompiledFunc
                         ,&background,0
                         ,&ThreadID);

  // start foreground jobs
  PrecompiledFunc(&foreground);
  InterpretedFunc(&interpreted);

  // wait for background job to finish
  WaitForSingleObject(h,1000); // Wait for background job to finish
  CloseHandle(h);
  DeleteCriticalSection(&c);
}

// Running 1 foreground and 2 background jobs //////////////////////////
int test2() {
  CRITICAL_SECTION c;
  InitializeCriticalSection(&c);

  // create data for jobs 
  FuncData background(100,"Compiled backgrond",&c);
  FuncData background2(100,"Compiled backgrond2",&c);
  FuncData interpreted(50,"Interpreted foreground",&c);
  FuncData foreground(50,"Compiled foreground",&c);

  // start background jobs
  DWORD ThreadID;
  HANDLE h[2];
  h[0]= CreateThread(NULL,0
                    ,(LPTHREAD_START_ROUTINE)PrecompiledFunc
                    ,&background,0
                    ,&ThreadID);

  h[1]= CreateThread(NULL,0
                    ,(LPTHREAD_START_ROUTINE)PrecompiledFunc
                    ,&background2,0
                    ,&ThreadID);

  // start foreground jobs
  PrecompiledFunc(&foreground);
  InterpretedFunc(&interpreted);

  // wait for background jobs to finish
  WaitForMultipleObjects(2,h,TRUE,1000); // Wait for background job to finish
  CloseHandle(h[0]);
  CloseHandle(h[1]);
  DeleteCriticalSection(&c);
}

// Test program main /////////////////////////////////////////////
int main() {
  test1();
  test2();
  return(0);
}
