// @(#)root/memstat:$Name$:$Id$
// Author: M.Ivanov -- Anar Manafov (A.Manafov@gsi.de) 28/04/2008

/*************************************************************************
 * Copyright (C) 1995-2008, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TH1.h"
// MemStat
#include "TMemStat.h"

const int g_count = 2;

void test1()
{
   double* x[g_count];
   for(int i = 0; i < g_count; ++i) {
      x[i] = new double[i+1];
   }
   for(int i = 0; i < g_count; ++i) {
      delete x[i];
   }
}
void test2()
{
   TH1F* h[100];
   for(int i = 0; i < 100; ++i) {
      h[i] = new TH1F(Form("h%d", i), "test", 10 + i, 0, 10);
      h[i]->Sumw2();
   }
}
void leak_test()
{
   TMemStat m("gnubuiltin");
   test1();
   test2();
}
