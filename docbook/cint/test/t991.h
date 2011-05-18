/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (cint@pcroot.cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/

#ifndef TOBJECT
#define TOBJECT

typedef int Option_t;

class TObject {
 public:
  TObject() { }
  TObject(const TObject& x) { }
  virtual ~TObject() { }
};

#endif
