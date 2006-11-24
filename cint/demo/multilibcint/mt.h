/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (cint@pcroot.cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
/************************************************************************
 * demo/multilibcint/mt.h 
 * 
 * Description:
 *  Cint's multi-thread workaround library. 
 *  Refer to README.txt in this directory.
 ************************************************************************/

#ifdef G__WIN32
#include <windows.h>
typedef HANDLE pthread_t;
#else
#include <pthread.h>
#endif

pthread_t G__createcintthread(char* args);
void G__joincintthread(pthread_t t1,int timeout=0);
void G__clearlibcint();

