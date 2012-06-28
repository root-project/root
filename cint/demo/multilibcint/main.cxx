/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (root-cint@cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
/************************************************************************
 * demo/multilibcint/main.c 
 * 
 * Description:
 *  Cint's multi-thread workaround demo program. 
 *  Refer to README.txt in this directory.
 ************************************************************************/

#include <stdio.h>
#ifdef G__WIN32
#include <windows.h>
#else
#include <unistd.h>
#include <pthread.h>
#endif

#ifdef __CINT__
#include "mt.dll"
#else
#include "mt.h"
#endif

/************************************************************************
 * test1() , Multi-thread demo program
 ************************************************************************/
int test1() {
  pthread_t t1,t2,t3,t4;
  // G__createcintthread copies $CINTSYSDIR/libcint.so as /tmp/libcint[0-9].so
  // and link them explicitly. Each thread owns one of libcintX.so which can
  // run simultaneously. This function runs cint as 'cint test1.cxx 10'
  t1 = G__createcintthread("test1.cxx 10");
  t2 = G__createcintthread("test2.cxx 7");
#ifdef G__WIN32
  Sleep(2000);
#else
  sleep(2);
#endif
  t3 = G__createcintthread("test1.cxx 10");
  t4 = G__createcintthread("test1.cxx 5");

  // Wait for all threads to finish.
  G__joincintthread(t1);
  printf("join t1\n");
  G__joincintthread(t2);
  printf("join t2\n");
  G__joincintthread(t3);
  printf("join t3\n");
  G__joincintthread(t4);
  printf("join t4\n");

  return(0);
}

/************************************************************************
 * test1()
 ************************************************************************/
int main() {
  // Running multi-thread demo functions twice.
  test1();
  test1();
  // Unlink and remove /tmp/libcint[0-9].so at the end of process
  G__clearlibcint();
  return 0;
}
