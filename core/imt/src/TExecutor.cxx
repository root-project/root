// @(#)root/thread:$Id$
// Author: Xavier Valls September 2020

/*************************************************************************
 * Copyright (C) 1995-2020, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/
#include "ROOT/TExecutor.hxx"

namespace ROOT{

namespace Internal{

//////////////////////////////////////////////////////////////////////////
/// \brief Return the number of pooled workers.
///
/// \return The number of workers in the pool in the executor used as a backend.

unsigned TExecutor::GetPoolSize() const
{
   unsigned poolSize{0u};
   switch(fExecPolicy){
      case ROOT::EExecutionPolicy::kSequential:
         poolSize = fSequentialExecutor->GetPoolSize();
         break;
      case ROOT::EExecutionPolicy::kMultiThread:
         poolSize = fThreadExecutor->GetPoolSize();
         break;
      case ROOT::EExecutionPolicy::kMultiProcess:
         poolSize = fProcessExecutor->GetPoolSize();
         break;
      default:
         break;
   }
   return poolSize;
}

} // namespace Internal
} // namespace ROOT
