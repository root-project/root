/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (cint@pcroot.cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
#ifndef T1247A
#define T1247A

#ifdef __CINT__
#pragma include_noerr "test1.dll"
#else
#include "t1247.h"
#endif

class B1 : public A1 {
 public:
};

class B2 : public A2 {
 public:
};

class B3 : public A3 {
 private:
  ~B3();
 public:
};

class B4 : public A4 {
 public:
};

class D1 : public C1 {
 public:
};

class D2 : public C2 {
 public:
};

class D3 : public C3 {
 public:
};

class D4 : public C4 {
 public:
};


#endif
