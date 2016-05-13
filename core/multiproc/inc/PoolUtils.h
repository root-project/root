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
#include "TList.h"
#include "TError.h"

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
  //////////////////////////////////////////////////////////////////////////
  /// Merge collection of TObjects.
  /// This functor looks for an implementation of the Merge method
  /// (e.g. TH1F::Merge) and calls it on the objects contained in objs.
  /// If Merge is not found, a null pointer is returned.
   template <class T>
   class ReduceObjects{
     public:
     T operator()(const std::vector<T> &objs){
      static_assert(std::is_constructible<TObject *, T>::value,
                    "The argument should be a vector of pointers to TObject or derived classes");
      if(objs.size() == 0)
         return nullptr;

      if(objs.size() == 1)
         return objs[0];

      //get first object from objs
      auto obj = objs[0];
      //get merge function
      ROOT::MergeFunc_t merge = obj->IsA()->GetMerge();
      if(!merge) {
         Error("PoolUtils::ReduceObjects", "could not find merge method for the TObject\n. Aborting operation.");
         return nullptr;
      }

      //put the rest of the objs in a list
      TList mergelist;
      unsigned NObjs = objs.size();
      for(unsigned i=1; i<NObjs; ++i) //skip first object
         mergelist.Add(objs[i]);

      //call merge
      merge(obj, &mergelist, nullptr);
      mergelist.Delete();

      //return result
      return obj;
    }
  };
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
