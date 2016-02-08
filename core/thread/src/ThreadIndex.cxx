#include "ThreadIndex.h"

#ifndef ROOT_ThreadLocalStorage
#include "ThreadLocalStorage.h"
#endif

namespace ROOT {
   namespace Internal {
      // A small generator of thread indices.
      // If not already available in the local storage, a map is queried for it
      // and the cache is built.
      unsigned ThreadIndexer::GetThreadIndex()
      {
         TTHREAD_TLS_DECL_ARG(unsigned, gThisThreadIndex, UINT_MAX);
         std::lock_guard<std::mutex> guard(*fThreadIndexerMutexPtr);

         if (UINT_MAX != gThisThreadIndex) {
            return gThisThreadIndex;
         }

         const auto id =  std::this_thread::get_id();

         auto keyValIt = fIDMap.find(id);
         if (keyValIt != fIDMap.end()) {
            gThisThreadIndex = keyValIt->second;
            return gThisThreadIndex;
         }
         gThisThreadIndex = fThreadIndex++;
         fIDMap.emplace(std::make_pair(id, gThisThreadIndex));
         return gThisThreadIndex;
      }
   }
}
