/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (cint@pcroot.cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
// Stub.h , interpreted function header

#ifndef STUB_H
#define STUB_H

double f1(int a,double b=1.41);

class StubBase {
 public:
  StubBase(short bin=0) { set(bin); }
  virtual ~StubBase() { }
  short b;
  short set(short bin);
  virtual int disp() ;
};

class StubClass : public StubBase {
 public:
  int a;
  double d;
  StubClass() { }
  StubClass(int ain,double din) { set(ain,din); }
  ~StubClass() { }
  double set(int ain,double din);
  int disp() ;
};

#endif
