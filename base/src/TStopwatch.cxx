// @(#)root/base:$Name:  $:$Id: TStopwatch.cxx,v 1.1.1.1 2000/05/16 17:00:39 rdm Exp $
// Author: Fons Rademakers   11/10/95

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TStopwatch                                                           //
//                                                                      //
// Stopwatch class. This class returns the real and cpu time between    //
// the start and stop events.                                           //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TStopwatch.h"
#include "TString.h"

#if defined(R__MAC)
#   include <time.h>
static clock_t gTicks = CLOCKS_PER_SEC;
#elif defined(R__UNIX)
#   include <sys/times.h>
#   include <unistd.h>
static clock_t gTicks = 0;
#elif defined(R__VMS)
#   include <time.h>
#   include <unistd.h>
static clock_t gTicks = 1000;
#elif defined(WIN32)
#include "TError.h"
    const Double_t gTicks = 1.0e-7;
#include "Windows4Root.h"
#endif


ClassImp(TStopwatch)

//______________________________________________________________________________
TStopwatch::TStopwatch()
{
   // Create a stopwatch and start it.

#ifdef R__UNIX
//printf("Tstopwatch constructor0 gTicks=%d\n",gTicks);
   if (!gTicks) gTicks = (clock_t)sysconf(_SC_CLK_TCK);
//printf("Tstopwatch constructor gTicks=%d\n",gTicks);
#endif
   fState         = kUndefined;
   fTotalCpuTime  = 0;
   fTotalRealTime = 0;
   fCounter       = 0;
   Start();
}

//______________________________________________________________________________
void TStopwatch::Start(Bool_t reset)
{
   // Start the stopwatch. If reset is kTRUE reset the stopwatch before
   // starting it (including the stopwatch counter).
   // Use kFALSE to continue timing after a Stop() without
   // resetting the stopwatch.

   if (reset) {
      fTotalCpuTime  = 0;
      fTotalRealTime = 0;
      fCounter       = 0;
   }
   if (fState != kRunning) {
#ifndef R__UNIX
      fStartRealTime = GetRealTime();
      fStartCpuTime  = GetCPUTime();
#else
      struct tms cpt;
      fStartRealTime = (Double_t)times(&cpt) / gTicks;
      fStartCpuTime  = (Double_t)(cpt.tms_utime+cpt.tms_stime) / gTicks;
#endif
   }
   fState = kRunning;
   fCounter++;
}

//______________________________________________________________________________
void TStopwatch::Stop()
{
   // Stop the stopwatch.

#ifndef R__UNIX
   fStopRealTime = GetRealTime();
   fStopCpuTime  = GetCPUTime();
#else
   struct tms cpt;
   fStopRealTime = (Double_t)times(&cpt) / gTicks;
   fStopCpuTime  = (Double_t)(cpt.tms_utime+cpt.tms_stime) / gTicks;
#endif
   if (fState == kRunning) {
      fTotalCpuTime  += fStopCpuTime  - fStartCpuTime;
      fTotalRealTime += fStopRealTime - fStartRealTime;
   }
   fState = kStopped;
}

//______________________________________________________________________________
void TStopwatch::Continue()
{
   // Resume a stopped stopwatch. The stopwatch continues counting from the last
   // Start() onwards (this is like the laptimer function).

   if (fState == kUndefined)
      Error("Continue", "stopwatch not started");

   if (fState == kStopped) {
      fTotalCpuTime  -= fStopCpuTime  - fStartCpuTime;
      fTotalRealTime -= fStopRealTime - fStartRealTime;
   }

   fState = kRunning;
}

//______________________________________________________________________________
Double_t TStopwatch::RealTime()
{
   // Return the realtime passed between the start and stop events. If the
   // stopwatch was still running stop it first.

   if (fState == kUndefined)
      Error("RealTime", "stopwatch not started");

   if (fState == kRunning)
      Stop();

   return fTotalRealTime;
}

//______________________________________________________________________________
Double_t TStopwatch::CpuTime()
{
   // Return the cputime passed between the start and stop events. If the
   // stopwatch was still running stop it first.

   if (fState == kUndefined)
      Error("RealTime", "stopwatch not started");

   if (fState == kRunning)
      Stop();

   return fTotalCpuTime;
}

//______________________________________________________________________________
Double_t TStopwatch::GetRealTime(){
#if defined(R__MAC)
   return(Double_t)clock() / gTicks;
#elif defined(R__UNIX)
   struct tms cpt;
   Double_t trt =  (Double_t)times(&cpt);
printf("GetRealTime: trt=%g, gTicks=%d\n",trt,gTicks);
   return trt / (double)gTicks;
#elif defined(R__VMS)
  return(Double_t)clock()/gTicks;
#elif defined(WIN32)
  union     {FILETIME ftFileTime;
             __int64  ftInt64;
            } ftRealTime; // time the process has spent in kernel mode
  SYSTEMTIME st;
  GetSystemTime(&st);
  SystemTimeToFileTime(&st,&ftRealTime.ftFileTime);
  return (Double_t)ftRealTime.ftInt64 * gTicks;
#endif
}

//______________________________________________________________________________
Double_t TStopwatch::GetCPUTime(){
#if defined(R__MAC)
   return(Double_t)clock() / gTicks;
#elif defined(R__UNIX)
   struct tms cpt;
   times(&cpt);
   return (Double_t)(cpt.tms_utime+cpt.tms_stime) / gTicks;
#elif defined(R__VMS)
   return(Double_t)clock()/gTicks;
#elif defined(WIN32)

  OSVERSIONINFO OsVersionInfo;

//*-*         Value                      Platform
//*-*  ----------------------------------------------------
//*-*  VER_PLATFORM_WIN32s          Win32s on Windows 3.1
//*-*  VER_PLATFORM_WIN32_WINDOWS       Win32 on Windows 95
//*-*  VER_PLATFORM_WIN32_NT            Windows NT
//*-*
  OsVersionInfo.dwOSVersionInfoSize=sizeof(OSVERSIONINFO);
  GetVersionEx(&OsVersionInfo);
  if (OsVersionInfo.dwPlatformId == VER_PLATFORM_WIN32_NT) {
    DWORD       ret;
    FILETIME    ftCreate,       // when the process was created
                ftExit;         // when the process exited

    union     {FILETIME ftFileTime;
               __int64  ftInt64;
              } ftKernel; // time the process has spent in kernel mode

    union     {FILETIME ftFileTime;
               __int64  ftInt64;
              } ftUser;   // time the process has spent in user mode

    HANDLE hProcess = GetCurrentProcess();
    ret = GetProcessTimes (hProcess, &ftCreate, &ftExit,
                                     &ftKernel.ftFileTime,
                                     &ftUser.ftFileTime);
    if (ret != TRUE){
      ret = GetLastError ();
      ::Error ("GetCPUTime", " Error on GetProcessTimes 0x%lx", (int)ret);
    }

    /*
     * Process times are returned in a 64-bit structure, as the number of
     * 100 nanosecond ticks since 1 January 1601.  User mode and kernel mode
     * times for this process are in separate 64-bit structures.
     * To convert to floating point seconds, we will:
     *
     *          Convert sum of high 32-bit quantities to 64-bit int
     */

      return (Double_t) (ftKernel.ftInt64 + ftUser.ftInt64) * gTicks;
  }
  else
      return GetRealTime();

#endif
}

//______________________________________________________________________________
void TStopwatch::Print(Option_t *)
{
   // Print the real and cpu time passed between the start and stop events.
   // and the number of times (slices) this TStopwatch was called
   // (if this number > 1)

   Double_t  realt = RealTime();

   Int_t  hours = Int_t(realt / 3600);
   realt -= hours * 3600;
   Int_t  min   = Int_t(realt / 60);
   realt -= min * 60;
   Int_t  sec   = Int_t(realt);
   Int_t counter = Counter();
   if (counter <= 1 )
      Printf("Real time %d:%d:%d, CP time %.3f", hours, min, sec, CpuTime());
   else
      Printf("Real time %d:%d:%d, CP time %.3f, %d slices", hours, min, sec, CpuTime(),counter);
}

