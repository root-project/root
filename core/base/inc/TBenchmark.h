// @(#)root/base:$Id$
// Author: Rene Brun   12/12/94

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/
// ---------------------------------- Benchmark.h

#ifndef ROOT_TBenchmark
#define ROOT_TBenchmark



//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TBenchmark                                                           //
//                                                                      //
// This class is a ROOT utility to help benchmarking applications       //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TNamed.h"
#include "TStopwatch.h"

class TBenchmark : public TNamed {

protected:

   Int_t      fNbench;          // Number of active benchmarks
   Int_t      fNmax;            // Maximum number of benchmarks initialized
   TString    *fNames;          //[fNbench] Names of benchmarks
   Float_t    *fRealTime;       //[fNbench] Real Time
   Float_t    *fCpuTime;        //[fNbench] Cpu Time
   TStopwatch *fTimer;          // Timers

   TBenchmark(const TBenchmark&);
   TBenchmark& operator=(const TBenchmark&);

public:
   TBenchmark();
   virtual            ~TBenchmark();
   Int_t              GetBench(const char *name) const;
   Float_t            GetCpuTime(const char *name);
   Float_t            GetRealTime(const char *name);
   virtual void       Print(Option_t *name="") const;
   virtual void       Reset();
   virtual void       Show(const char *name);
   virtual void       Start(const char *name);
   virtual void       Stop(const char *name);
   virtual void       Summary(Float_t &rt, Float_t &cp);

   ClassDef(TBenchmark,0)  //ROOT utility to help benchmarking applications
};

R__EXTERN TBenchmark  *gBenchmark;

#endif
