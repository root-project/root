/* @(#)root/multiproc:$Id$ */
// Author: Enrico Guiraud July 2015

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_PoolCode
#define ROOT_PoolCode

namespace PoolCode {

   //////////////////////////////////////////////////////////////////////////
   ///
   /// An enumeration of the message codes handled by TPool and TPoolWorker.
   ///
   //////////////////////////////////////////////////////////////////////////
   
   enum EPoolCode : unsigned {
      kExecFunc = 0,    ///< Execute function without arguments
      kExecFuncWithArg, ///< Execute function with the argument contained in the message
      kFuncResult,      ///< The message contains the result of a function execution
      kIdling,          ///< We are ready for the next task
      kSendResult       ///< Ask for a kFuncResult
   };

}

#endif
