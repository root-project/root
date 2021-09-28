// @(#)root/eve7:$Id$
// Authors: Matevz Tadel & Alja Mrak-Tadel: 2006, 2007, 2018

/*************************************************************************
 * Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT7_REveSystem
#define ROOT7_REveSystem

#include "TSystem.h"
#include <ctime>

#ifdef _MSC_VER
   typedef int pid_t;
#endif

namespace ROOT {
namespace Experimental {

// Have to be PODs so we can mem-copy them around or send them via IPC or network.

struct REveServerStatus
{
   pid_t         fPid = 0;
   int           fNConnects = 0;
   int           fNDisconnects = 0;
   std::time_t   fTStart = 0;
   std::time_t   fTLastMir = 0;
   std::time_t   fTLastConnect = 0;
   std::time_t   fTLastDisconnect = 0;
   ProcInfo_t    fProcInfo; // To be complemented with cpu1/5/15 and memgrowth1/5/15 on the collector side.
#if defined(R__LINUX) || defined(_MSC_VER)
   std::timespec fTReport = {0, 0};
#else
   std::time_t   fTReport = 0;
#endif
   int n_active_connections() const { return fNConnects - fNDisconnects; }
};

}}

#endif
