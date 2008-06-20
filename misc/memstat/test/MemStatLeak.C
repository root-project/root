// @(#)root/memstat:$Name$:$Id: testMemStat.C 143 2008-06-17 16:15:42Z anar $
// Author: M.Ivanov -- Anar Manafov (A.Manafov@gsi.de) 28/04/2008

/*************************************************************************
 * Copyright (C) 1995-2008, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// An example script, which "leaks" and TMemStat is used the leaks to   //
// detect.                                                              //
// This script is called from the "MemStat.C" script, which shows how   //
// to print results and call TMemStat's graphics user interface.        //
//                                                                      //
// Please use the "MemStat.C" script to start the demo.                 //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

// ROOT
#include "TSystem.h"
#include "TStopwatch.h"
#include "TTimeStamp.h"
#include "TRandom.h"
#include "TArrayF.h"
#include "TArrayD.h"
#include "TArrayI.h"
#include "TArrayS.h"
#include "TNamed.h"
#include "TObjString.h"
#include "TVectorD.h"
// MemStat
#include "TMemStat.h"

void funF();
void funD();
void funS();
void funI();
void funA();
void funObject();

const Int_t nloop = 50000;
const Int_t nskip = 100;

void MemStatLeak()
{
   // Creating the TMemStat object
   TMemStat m("new,gnubuildin");
   TStopwatch stopwatch;
   stopwatch.Start();
   TTimeStamp stamp;
   // adding user's step Start
   m.AddStamp("Start");
   funF();
   // adding user's step funF
   m.AddStamp("funF");
   funD();
   m.AddStamp("funD");
   funS();
   m.AddStamp("funS");
   funI();
   m.AddStamp("funI");
   funA();
   m.AddStamp("funA");
   funObject();
   m.AddStamp("funObject");
   stopwatch.Stop();
   stopwatch.Print();
}

// the following functions are "leaking" ones...
void funF()
{
   for (Int_t i = 0; i < nloop; ++i) {
      TArrayF * arr = new TArrayF(100);
      if (i % nskip == 0)
         continue;
      delete arr;
   }
}

void funD()
{
   for (Int_t i = 0; i < nloop; ++i) {
      TArrayD * arr = new TArrayD(100);
      if (i % nskip == 0)
         continue;
      delete arr;
   }
}

void funS()
{
   for (Int_t i = 0; i < nloop; ++i) {
      TArrayS * arr = new TArrayS(100);
      if (i % nskip == 0)
         continue;
      delete arr;
   }
}

void funI()
{
   for (Int_t i = 0; i < nloop; ++i) {
      TArrayI * arr = new TArrayI(100);
      if (i % nskip == 0)
         continue;
      delete arr;
   }
}


void funA()
{
   funD();
   funF();
   funI();
   funS();
}

void funObject()
{
   TObjArray arr(2000);
   for (Int_t i = 0; i < 2000; ++i) {
      TObject * o = new TNamed(Form("xxx%d", i), "leak");
      if (i % nskip > 4)
         arr.AddAt(o, i);
   }
   for (Int_t i = 0; i < 2000; ++i) {
      if (arr.At(i))
         delete arr.At(i);
   }
}
