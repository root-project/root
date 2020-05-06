#include "tbb/task_arena.h"
namespace ROOT {
    namespace Internal {
      // Wrapper for tbb::task_arena.
      //
      //tbb::task_arena is an alias of tbb::interface7::task_arena, which doesn't allow
      // to forward declare tbb::task_arena. To avoid code breaking on tbb interface changes
      // we don't forward declare tbb::interface7::task_arena and instead we wrap tbb::task_arena
      // in the forward declared class RArena
      class RArena {
      public:
        /// Access the wrapped object and allow to call its methods.
        tbb::task_arena *operator->() {
            return fTBBArena.get();
        }
      private:
         std::unique_ptr<tbb::task_arena> fTBBArena{new tbb::task_arena()};
      };
    }
}
