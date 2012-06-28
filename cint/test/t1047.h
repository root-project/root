/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (root-cint@cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/


// test.H
#include <string>
using namespace std;
 
namespace test {
   extern char const * const default_str;
   extern const char * const default_strx;
   extern string const abc;
}

namespace test {
   char const * const default_str = "default_str";
   const char * const default_strx = "default_strx";
   string const abc="abc";
}


// testLinkDef.h
#ifdef __CINT__

#pragma link off all globals;
#pragma link off all classes;
#pragma link off all functions;

#pragma link C++ namespace test;
#pragma link C++ nestedclass;
#pragma link C++ nestedtypedef;
#endif



