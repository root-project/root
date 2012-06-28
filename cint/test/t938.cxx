/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (root-cint@cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
#include <stdio.h>

template <class T>
class container {
public:
  container(int n,const T& value=T()) ;
  container() { }
};

namespace MySpace {
   template <class T>
   class wrapper {
   public:
     wrapper(){}
   };

   class sky {
   public:
     sky() { }
#ifndef SECOND
     container< MySpace::wrapper<float> > cont; //!
#endif
      container< ::MySpace::wrapper<float> > cont2; //!
   };

}

int main() { 
  printf("t938.cxx is read without problem. \n");
  return 0;
}

/*

There are 2 problems.  To see the second problem I commented out the 
line defining 'cont' after the first .L.

cint : C/C++ interpreter  (mailing list 'root-cint@cern.ch')
   Copyright(c) : 1995~2002 Masaharu Goto 
   revision     : 5.15.61, Oct 6 2002 by M.Goto

No main() function found in given source file. Interactive interface started.
'?' for help, '.q' to quit, '{ statements }' or '.p [expression]' to evaluate

cint> .L pb01.C
Error: Function wrapper<float>() is not defined in current scope  FILE:pb01.C LINE:4
Possible candidates are...
filename       line:size busy function type and name  
*** Interpreter error recovered ***
cint>  .L pb01.C
Error: no such template ::MySpace::wrapper<float> FILE:pb01.C LINE:18
*** Interpreter error recovered ***
cint> .q
  Bye... (try 'qqq' if still running)

*/
