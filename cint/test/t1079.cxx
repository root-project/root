/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (root-cint@cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/

#include <cstdio>

#if defined(interp) && defined(makecint)
#pragma include "test.dll"
#else
#include "t1079.h"
#endif

using namespace std;

template<class T> 
void f(T& x) {
  smart_ptr<T> p(&x);
  std::printf("%s\n",p->c_str());
}

int main() {
  std::string x("stringx");
  f(x);
  String y("Stringy");
  f(y);
  return 0;
}
