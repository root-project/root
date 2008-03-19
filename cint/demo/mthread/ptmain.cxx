/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (cint@pcroot.cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/

// Linux2.0 egcs : g++ -lpthread pthread1.cxx
// HP-UX aCC     : aCC -D_POSIX_C_SOURCE=199506L -lpthread pthread1.cxx


#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include "thread.dll"

#if 0
void *thread2(void* arg) {
  int n = 1000000;
  for(int i=0;i<10*n;i++) {
    if(i%n==0) {
      printf("%s i=%d\n",(char*)arg,i);
    }
  }
  return(0);
}
#endif

pthread_t test(int n) {
  pthread_t thread;
  pthread_attr_t attr;
  void *arg = (void*)"sub thread";

  int stat = pthread_create(&thread,NULL,thread1,arg);
  printf("stat=%d\n",stat);
  //if(stat==EAGAIN) printf("EAGAIN\n");
  //if(stat==EPARM) printf("EPARM\n");
  //if(stat==EINVAL) printf("EINVAL\n");
  return(thread);
}

int main(int argc, char** argv) {
  int n=100;
  if(argc>1) n=atoi(argv[1]);
  thread1((void*)"main thread1");
  pthread_t pthread = test(n);
  thread1((void*)"main thread2");
  pthread_join(pthread,0);
  return(0);
}


