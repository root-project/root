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

#include <atomic>
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

static std::atomic_int &GetParBranchProcessingCount()
{
   static std::atomic_int count(0);
   return count;
}

static std::atomic_int &GetParTreeProcessingCount()
{
   static std::atomic_int count(0);
   return count;
}

static UInt_t &GetImplicitMTPoolSize()
{
   static UInt_t size = 0;
   return size;
};

extern "C" void ROOT_TImplicitMT_EnableImplicitMT(UInt_t numthreads)
{
   if (!GetImplicitMTFlag()) {
      if (!GetScheduler().is_active()) {
         TThread::Initialize();
         bool defaultSize = numthreads == 0;
         if (defaultSize)
            numthreads = tbb::task_scheduler_init::automatic;

         GetScheduler().initialize(numthreads);
         GetImplicitMTPoolSize() = defaultSize ? tbb::task_scheduler_init::default_num_threads() : numthreads;
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

extern "C" UInt_t ROOT_TImplicitMT_GetImplicitMTPoolSize()
{
   return GetImplicitMTPoolSize();
};


extern "C" void ROOT_TImplicitMT_EnableParBranchProcessing()
{
   ++GetParBranchProcessingCount();
};

extern "C" void ROOT_TImplicitMT_DisableParBranchProcessing()
{
   --GetParBranchProcessingCount();
};

extern "C" bool ROOT_TImplicitMT_IsParBranchProcessingEnabled()
{
   return GetParBranchProcessingCount() > 0;
};

extern "C" void ROOT_TImplicitMT_EnableParTreeProcessing()
{
   ++GetParTreeProcessingCount();
};

extern "C" void ROOT_TImplicitMT_DisableParTreeProcessing()
{
   --GetParTreeProcessingCount();
};

extern "C" bool ROOT_TImplicitMT_IsParTreeProcessingEnabled()
{
   return GetParTreeProcessingCount() > 0;
};
