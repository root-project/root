// @(#)root/thread:$Id$
// // Author: Enric Tejedor Saavedra   03/12/15
//
/*************************************************************************
 * Copyright (C) 1995-2015, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TImplicitMT                                                          //
//                                                                      //
// This file implements the methods to enable, disable and check the    //
// status of the global implicit multi-threading in ROOT.               //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TError.h"
#include "TThread.h"

#include "tbb/task_scheduler_init.h"


static tbb::task_scheduler_init &GetScheduler()
{
   static tbb::task_scheduler_init scheduler(tbb::task_scheduler_init::deferred);
   return scheduler;
}

static bool &GetImplicitMTFlag()
{
   static bool enabled = false;
   return enabled;
}

static bool &GetParBranchProcessingFlag()
{
   static bool enabled = false;
   return enabled;
}

static bool &GetParTreeProcessingFlag()
{
   static bool enabled = false;
   return enabled;
}

extern "C" void ROOT_TImplicitMT_EnableImplicitMT(UInt_t numthreads)
{
   if (!GetImplicitMTFlag()) {
      if (!GetScheduler().is_active()) {
         TThread::Initialize();

         if (numthreads == 0)
            numthreads = tbb::task_scheduler_init::automatic;

         GetScheduler().initialize(numthreads);
      }
      GetImplicitMTFlag() = true;
   }
   else {
      ::Warning("ROOT_TImplicitMT_EnableImplicitMT", "Implicit multi-threading is already enabled");
   }
};

extern "C" void ROOT_TImplicitMT_DisableImplicitMT()
{
   if (GetImplicitMTFlag()) {
      GetImplicitMTFlag() = false;
   }
   else {
      ::Warning("ROOT_TImplicitMT_DisableImplicitMT", "Implicit multi-threading is already disabled");
   }
};

extern "C" bool ROOT_TImplicitMT_IsImplicitMTEnabled()
{
   return GetImplicitMTFlag();
};

extern "C" void ROOT_TImplicitMT_EnableParBranchProcessing()
{
   GetParBranchProcessingFlag() = true;
};

extern "C" bool ROOT_TImplicitMT_IsParBranchProcessingEnabled()
{
   return GetParBranchProcessingFlag();
};

extern "C" void ROOT_TImplicitMT_EnableParTreeProcessing()
{
   GetParTreeProcessingFlag() = true;
};

extern "C" bool ROOT_TImplicitMT_IsParTreeProcessingEnabled()
{
   return GetParTreeProcessingFlag();
};
