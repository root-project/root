/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (cint@pcroot.cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/

//#define T928A
#include "t928.h"

class C {
 public:
};


#ifdef __MAKECINT__
#pragma link off defined_in ../t928.h;

//#pragma link off all_function A<int>;
//#pragma link off all_datamember A<int>;
//#pragma link off all_method A<short>;
//#pragma link off all_datamember A<short>;

#pragma link C++ function A<int>::f<C>;
#pragma link C++ function A<int>::f<float>;
#pragma link C++ function A<short>::f<C>();
#pragma link C++ function A<short>::f<double>();
#pragma link C++ function A<int>::g(C);
#pragma link C++ function A<int>::g(double);
#endif
