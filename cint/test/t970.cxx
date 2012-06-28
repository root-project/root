/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (root-cint@cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
// 021218loop0-2.txt,  related with t964.cxx(root)

#ifdef __CINT__
#pragma include "test.dll"
#else
#include "t970.h"
#endif

int main()
{
   // --
#ifndef CINT_HIDE_FAILURE
   TVector indice(4);
   for (int i = 2; i < 7; i++) {
      if (i != 2) {
         // --
#ifdef __CINT__
         // The cint failure to handle scope properly makes the destructor
         // call for the x object happen at the wrong time, fake it with
         // temporaries.
         // FIXME: This also doesn't work now that temps live longer.
         TMatrixRow(i) = indice;
#else // __CINT__
         TMatrixRow x(i);
         x = indice;
#endif // __CINT__
         // --
      }
   }
#endif // CINT_HIDE_FAILURE
   return  0;
}
