/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (root-cint@cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/


#include <vector>
#ifndef __hpux
using namespace std;
#endif

typedef int Int_t;

#ifdef __MAKECINT__
#pragma link C++ class vector<vector< Int_t> >;
#endif
