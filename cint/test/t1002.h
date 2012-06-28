/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (root-cint@cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/

// file Test.h
#include <stdio.h>

template <typename R> 
class Marshal {
 public:
  Marshal() { printf("Marshhal<T>()\n"); }
  ~Marshal() { printf("~Marshhal<T>()\n"); }
};
  
template <> 
class Marshal<int>
{
 public:
  Marshal() { printf("Marshhal<int>()\n"); }
  ~Marshal() { printf("~Marshhal<int>()\n"); }
};

#ifdef __MAKECINT__
#pragma link C++ class Marshal<double>;
#pragma link C++ class Marshal<int>;
#endif

