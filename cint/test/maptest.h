/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (cint@pcroot.cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
#include <map>
#include <string>

#ifndef __hpux
using namespace std;
#endif


#ifdef __MAKECINT__
#pragma link C++ class map<string,string>;
#pragma link C++ class map<string,string>::iterator;
#endif
