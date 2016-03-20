// @(#)root/gl:$Id$
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

/** \class TGLStopwatch
\ingroup opengl
Stopwatch object for timing GL work. We do not use the TStopwatch as
we need to perform GL flushing to get accurate times + we record
timing overheads here.
*/

ClassImp(TGLStopwatch);

////////////////////////////////////////////////////////////////////////////////
/// Construct stopwatch object.

TGLStopwatch::TGLStopwatch() : fStart(0), fEnd(0), fLastRun(0)
{
}

////////////////////////////////////////////////////////////////////////////////
/// Destroy stopwatch object.

TGLStopwatch::~TGLStopwatch()
{
}

////////////////////////////////////////////////////////////////////////////////
/// Start timing.

void TGLStopwatch::Start()
{
   fStart   = GetClock();
   fEnd     = 0;
}

// In milliseconds
////////////////////////////////////////////////////////////////////////////////
/// Return lap time since Start(), in milliseconds.

Double_t TGLStopwatch::Lap() const
{
   if (fStart == 0)
      return 0;
   else
      return GetClock() - fStart;
}

// In milliseconds
////////////////////////////////////////////////////////////////////////////////
/// End timing, return total time since Start(), in milliseconds.

Double_t TGLStopwatch::End()
{
   if (fStart == 0)
      return 0;
   if (fEnd == 0) {
      fEnd = GetClock();
      fLastRun = fEnd - fStart;
   }
   return fLastRun;
}

////////////////////////////////////////////////////////////////////////////////
/// Get internal clock time, in milliseconds.

Double_t TGLStopwatch::GetClock(void) const
{
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
