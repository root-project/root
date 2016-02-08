#ifndef ROOT_ThreadIndex
#define ROOT_ThreadIndex

#ifndef ROOT_ThreadLocalStorage
#include "ThreadLocalStorage.h"
#endif

#include <map>
#include <mutex>
#include <thread>
#include <climits>

namespace ROOT {

   namespace Internal {

      /// Get the index of the current thread
      class ThreadIndexer {
      private:
         unsigned fThreadIndex = 0U;
         std::mutex *fThreadIndexerMutexPtr = new std::mutex();
         std::map<std::thread::id, unsigned> fIDMap;
      public:
         unsigned GetThreadIndex() {
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
      };


   }
}

#endif
