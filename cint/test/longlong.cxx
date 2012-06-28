/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (root-cint@cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
#include <stdio.h>
#include <iostream>
typedef long long Long64_t;
int main()
{
   Long64_t *w=new Long64_t[5];
   Long64_t x=999741748;
   Long64_t y=162909875;
   w[0]=x<<31;
   w[0]+=y;
   std::cout << "cout:        " << w[0] << std::endl;
   printf("printf ll:   %lld\n", w[0]);
   printf("one less ll: %lld\n", 2146929056215846578LL);
   printf("again ll:    %lld\n", w[0]);
   // warnings are intentional
   printf("printf l:    %ld\n", w[0]);
   printf("one less l:  %ld\n", 2146929056215846578LL);
   printf("again l:     %ld\n", w[0]);
   
   delete []w;
      
  return 0;
}
