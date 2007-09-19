// @(#)root/base:$Id$
// Author: Fons Rademakers   11/10/95

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TStopwatch
#define ROOT_TStopwatch


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TStopwatch                                                           //
//                                                                      //
// Stopwatch class. This class returns the real and cpu time between    //
// the start and stop events.                                           //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TObject
#include "TObject.h"
#endif


class TStopwatch : public TObject {

private:
   enum EState { kUndefined, kStopped, kRunning };

   Double_t     fStartRealTime;   //wall clock start time
   Double_t     fStopRealTime;    //wall clock stop time
   Double_t     fStartCpuTime;    //cpu start time
   Double_t     fStopCpuTime;     //cpu stop time
   Double_t     fTotalCpuTime;    //total cpu time
   Double_t     fTotalRealTime;   //total real time
   EState       fState;           //stopwatch state
   Int_t        fCounter;         //number of times the stopwatch was started

   static Double_t GetRealTime();
   static Double_t GetCPUTime();

public:
   TStopwatch();
   void        Start(Bool_t reset = kTRUE);
   void        Stop();
   void        Continue();
   Int_t       Counter() const { return fCounter; }
   Double_t    RealTime();
   void        Reset() { ResetCpuTime(); ResetRealTime(); }
   void        ResetCpuTime(Double_t time = 0) { Stop();  fTotalCpuTime = time; }
   void        ResetRealTime(Double_t time = 0) { Stop(); fTotalRealTime = time; }
   Double_t    CpuTime();
   void        Print(Option_t *option="") const;

   ClassDef(TStopwatch,1)  //A stopwatch which times real and cpu time
};

#endif
