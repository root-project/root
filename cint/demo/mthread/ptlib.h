/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (root-cint@cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
// ptlib.h  : Multi thread demo program using pthread
//
//  Usage:
//   makecint -mk makethread -dl thread.dll -H ptlib.h
//   make -f makethread
//   cint ptmain.cxx

#include <stdio.h>
#include <stdlib.h>
//#include <pthread.h>

void *thread1(void* arg) {
  int n = 1000000;
  for(int i=0;i<10*n;i++) {
    if(i%n==0) {
      printf("%s i=%d\n",(char*)arg,i);
    }
  }
  return(0);
}
