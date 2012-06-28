/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (root-cint@cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/

#ifndef __CINT__
#include <stdlib.h>
#include <stdio.h>
#endif //__CINT__
class aaa1 {
 public:
  char c1;
  void fun1(){printf("fun1\n");}
};
class aaa2 {
 public:
  void fun2(){printf("fun2\n");}
  char c2;
};

class aaa 
#if 0
:public aaa1,public aaa2 
#else
:private aaa1,private aaa2 
#endif
{
 public:
  void fun(){printf("fun\n");}
  char c;
};
#ifdef __CINT__
#pragma link off all globals;
#pragma link off all classes;
#pragma link off all functions;
#pragma link C++ class aaa+;
#pragma link C++ class aaa1+;
#pragma link C++ class aaa2+;
#endif //__CINT__


