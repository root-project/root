/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (root-cint@cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/

#include <iostream>

class xclass {
 public:
  void f(int open_mode = std::ios::in);
};
namespace myspace {
   class xclass {
   public:
       void f(int open_mode = std::ios::in);
   };
   class myclass {
   public:
       myclass(const char *name, int open_mode = std::ios::in);
   };
}

