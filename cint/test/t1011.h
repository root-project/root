/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (root-cint@cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
// 030211defaulttmppara

#include <iostream>
using namespace std;

namespace NS {

template <typename R> class Marshal {
 public:
  Marshal() {}
  ~Marshal() {}
};

template <class R,class Marsh=Marshal<R> > class Something {
 public:
  Something() {}
  ~Something() {}
  void no_op() { std::cout << "fully templated version\n"; }
};

}

//using namespace NS;

class MyClass {
 public:
  MyClass() {}
  ~MyClass() {}
  NS::Something<int> something;
};

#ifdef __MAKECINT__
#pragma link C++ nestedclasses;
#pragma link C++ nestedtypedefs;
#pragma link C++ namespace NS;
#pragma link C++ class NS::Marshal<int>;
#pragma link C++ class MyClass;
#endif

