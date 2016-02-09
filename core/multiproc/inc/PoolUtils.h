/* @(#)root/multiproc:$Id$ */
// Author: Enrico Guiraud July 2015

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_PoolUtils
#define ROOT_PoolUtils

#include "TObject.h"
#include <vector>

namespace PoolCode {

   //////////////////////////////////////////////////////////////////////////
   ///
   /// An enumeration of the message codes handled by TPool, TPoolWorker and TPoolProcessor
   ///
   //////////////////////////////////////////////////////////////////////////

   enum EPoolCode : unsigned {
      //not an enum class because we want to be able to easily cast back and forth from unsigned
      /* TPool::Map */
      kExecFunc = 0,    ///< Execute function without arguments
      kExecFuncWithArg, ///< Execute function with the argument contained in the message
      kFuncResult,      ///< The message contains the result of a function execution
      /* TPool::MapReduce */
      kIdling,          ///< We are ready for the next task
      kSendResult,      ///< Ask for a kFuncResult/kProcResult
      /* TPool::Process */
      kProcFile,        ///< Tell a TPoolProcessor which tree to process. The object sent is a TreeInfo
      kProcRange,       ///< Tell a TPoolProcessor which tree and entries range to process. The object sent is a TreeRangeInfo
      kProcTree,        ///< Tell a TPoolProcessor to process the tree that was passed to it at construction time
      kProcResult,      ///< The message contains the result of the processing of a TTree
      kProcEnded,       ///< Tell the client we are done processing (i.e. we have reached the target number of entries to process)
      kProcError,       ///< Tell the client there was an error while processing
   };

}

//////////////////////////////////////////////////////////////////////////
///
/// This namespace contains pre-defined functions to be used in
/// conjuction with TPool::Map and TPool::MapReduce.
///
//////////////////////////////////////////////////////////////////////////
namespace PoolUtils {
   TObject *ReduceObjects(const std::vector<TObject *> &objs);
}

namespace ROOT {
   namespace Internal {
      namespace PoolUtils {
         // The caster casts a pointer to a TObject to a specialised type F and leaves
         // unaltered the other cases.
         template <class O, class F>
         class ResultCaster {
         public:
            static O CastIfNeeded(O &&obj)
            {
               return obj;
            }
         };
         template <class F>
         class ResultCaster<TObject *, F> {
         public:
            static typename std::enable_if<std::is_pointer<F>::value, F>::type CastIfNeeded(TObject *obj)
            {
               return static_cast<F>(obj);
            }
         };
      }
   }
}


#endif
