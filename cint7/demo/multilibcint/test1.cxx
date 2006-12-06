/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (cint@pcroot.cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
// Test program that is run by multi-thread libcint

// unistd.h loads posix.dll. posix.dll must be compiled 
#ifdef WIN32
#include <windows.h>
#else
#include <unistd.h>
#endif

int main(int argc,char** argv) {
  int n;
  printf("pid=%d started\n",getpid());
  if(argc>1) n = atoi(argv[1]);
  else       n = 10;
  for(int i=0;i<n;i++) {
#ifdef WIN32
    Sleep(1000);
#else
    sleep(1);
#endif
    printf("test i=%d pid=%d\n",i,getpid());
  }
  printf("pid=%d finished\n",getpid());
  return(0);
}
