#include "tbb/task_arena.h"

namespace ROOT {
class ROpaqueTaskArena : public tbb::task_arena {
public:
   using tbb::task_arena::task_arena;
};
}
