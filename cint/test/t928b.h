/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (cint@pcroot.cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/

#include "t928.h"

class D {
 public:
};


#ifdef __MAKECINT__
#pragma link off defined_in ../t928.h;

//#pragma link off all_function A<int>;
//#pragma link off all_datamember A<int>;
//#pragma link off all_method A<short>;
//#pragma link off all_datamember A<short>;

#pragma link C++ function A<int>::f<D>;
#pragma link C++ function A<short>::f<D>();
#pragma link C++ function A<int>::g(D);
#endif
