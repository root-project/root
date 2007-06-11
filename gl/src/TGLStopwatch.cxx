// @(#)root/gl:$Name:  $:$Id: TGLStopwatch.cxx,v 1.1.1.1 2007/04/04 16:01:44 mtadel Exp $
// Author:  Richard Maunder  25/05/2005

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TGLStopwatch.h"
#include "TGLIncludes.h"

#ifdef R__WIN32
#include <Windows.h>  // For GetSystemTimeAsFileTime()
#else
#include <sys/time.h> // For gettimeofday()
#endif

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGLStopwatch                                                         //
//                                                                      //
// Stopwatch object for timing GL work. We do not use the TStopwatch as //
// we need to perform GL flushing to get accurate times + we record     //
// timing overheads here.                                               //
//////////////////////////////////////////////////////////////////////////

ClassImp(TGLStopwatch)

Bool_t   TGLStopwatch::fgInitOverhead = kFALSE;
Double_t TGLStopwatch::fgOverhead = 0.0;

//______________________________________________________________________________
TGLStopwatch::TGLStopwatch()
{
   // Construct stopwatch object, initialising timing overheads if not done.

   // MT: bypass
   // if (!fgInitOverhead) InitOverhead();
}

//______________________________________________________________________________
TGLStopwatch::~TGLStopwatch()
{
   // Destroy stopwatch object
}

//______________________________________________________________________________
void TGLStopwatch::Start()
{
   // Start timing.

   FinishDrawing();
   fStart = WaitForTick();
}

// In milliseconds
//______________________________________________________________________________
Double_t TGLStopwatch::Lap() const
{
   // Return lap time since Start(), in milliseconds.

   FinishDrawing();
   Double_t elapsed = GetClock() - fStart - fgOverhead;
   return elapsed > 0.0 ? elapsed : 0.0;
}

// In milliseconds
//______________________________________________________________________________
Double_t TGLStopwatch::End()
{
   // End timing, return total time since Start(), in milliseconds.

   return Lap();
}

// In milliseconds
//______________________________________________________________________________
Double_t TGLStopwatch::GetClock(void) const
{
   // Get internal clock time, in milliseconds.

#ifdef R__WIN32
   // Use performance counter (system dependent support) if possible
   static LARGE_INTEGER perfFreq;
   static Bool_t usePerformanceCounter = QueryPerformanceFrequency(&perfFreq);

   if (usePerformanceCounter) {
      LARGE_INTEGER counter;
      QueryPerformanceCounter(&counter);
      Double_t time = static_cast<Double_t>(counter.QuadPart)*1000.0 /
                      static_cast<Double_t>(perfFreq.QuadPart);
      return time;
   }

   // TODO: Portability - check with Rene
   FILETIME        ft;
   ULARGE_INTEGER  uli;
   __int64         t;

   GetSystemTimeAsFileTime(&ft);
   uli.LowPart  = ft.dwLowDateTime;
   uli.HighPart = ft.dwHighDateTime;
   t  = uli.QuadPart;                        // 100-nanosecond resolution
   return static_cast<Double_t>(t /= 1E4);   // Milliseconds
#else
   struct timeval tv;
   gettimeofday(&tv, 0);
   return static_cast<Double_t>(tv.tv_sec*1E3) + static_cast<Double_t>(tv.tv_usec) / 1E3;
#endif
}

//______________________________________________________________________________
void TGLStopwatch::FinishDrawing(void) const
{
   // Force completion of GL drawing.

   // MT bypass:
   // glFinish();
}

//______________________________________________________________________________
Double_t TGLStopwatch::WaitForTick(void)  const
{
   // Wait for next clock increment - return it in milliseconds

   // MT bypass:
   return GetClock();

   Double_t start;
   Double_t current;

   start = GetClock();

   // Next tick
   while ((current = GetClock()) == start);

   return current;
}

//______________________________________________________________________________
void TGLStopwatch::InitOverhead(void) const
{
   // Calcualte timing overhead.

   // MT bypass (now not even called):
   fgInitOverhead = kTRUE;
   fgOverhead     = 0.0;

   Double_t runTime;
   Long_t   reps;
   Double_t start;
   Double_t finish;
   Double_t current;

   start = GetClock();

   // Next tick
   while ((finish = GetClock()) == start);

   // Test on 100 ticks range to 0.1 sec - 0.5 sec
   runTime = 100.0 * (finish - start);
   if (runTime < 100)
      runTime = 100;
   else if (runTime > 500)
      runTime = 1000;

   // Clear GL pipe
   FinishDrawing();

   // Overhead for finalization and timing routines
   reps = 0;
   start = WaitForTick();
   finish = start + runTime;
   do {
      FinishDrawing();
      ++reps;
   } while ((current = GetClock()) < finish);

   fgOverhead = (current - start) / (Double_t) reps;
   fgInitOverhead = true;
}
