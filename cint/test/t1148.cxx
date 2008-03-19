/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (cint@pcroot.cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/

#ifndef FOO
#define FOO

#include <stdio.h>

#ifdef __cplusplus 
extern "C" {
  void fff(char *foobar) { printf("fff(%s)\n",foobar); } 
}
#endif

#ifdef __cplusplus 
extern "C" {  int xxx; }
#endif

#ifdef __cplusplus 
extern "C" { 
#endif   

  void aaa(char *foobar) { printf("aaa(%s)\n",foobar); } 

#ifdef __cplusplus 
extern "C" { 
#endif   

  void foo(char *foobar); 

#ifdef __cplusplus
}
#endif

  void ccc(char *foobar) { printf("ccc(%s)\n",foobar); } 

#ifdef __cplusplus
}
#endif

  void ddd(char *foobar) { printf("ddd(%s)\n",foobar); } 

#endif

#if defined(__cplusplus)
extern "C" { 
#endif   
#if defined(__cplusplus)
extern "C" { 
#endif   

  void foo(char *foobar) {printf("foo(%s)\n",foobar);}

#if defined(__cplusplus)
}
#endif
#if defined(__cplusplus)
}
#endif

int main() {
  foo("abc");
  aaa("abc");
  ccc("abc");
  ddd("abc");
  fff("abc");
  return 0;
}
