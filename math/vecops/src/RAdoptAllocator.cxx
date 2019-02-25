#include <ROOT/RAdoptAllocator.hxx>
#include <Rtypes.h>

// All this code a workaround. It is needed because cling
// cannot handle thread_local yet.

const unsigned int gBuffersNumber = 32U;

namespace ROOT {
namespace Internal {
namespace VecOps {

// This wrapper allows to deallocate the memory at tear down
class RBufferStackWrapper {
   using StdAlloc_t = std::allocator<char>;
   std::stack<void *> fBufStack;
   const std::size_t fBuffersSize;
   std::allocator<char> fCharAlloc;

public:
   RBufferStackWrapper(std::size_t typeSize)
      : fBuffersSize(::ROOT::Internal::VecOps::gBuffersSize * typeSize * typeSize / sizeof(char))
   {
      char *p;
      for (auto i = 0U; i < gBuffersNumber; ++i) {
         p = std::allocator_traits<StdAlloc_t>::allocate(fCharAlloc, fBuffersSize);
         fBufStack.push((void*)p);
      }
   }

   std::stack<void *> *Get() { return &fBufStack; }

   ~RBufferStackWrapper()
   {
      while (!fBufStack.empty()) {
         auto p = (char *) fBufStack.top();
         std::allocator_traits<StdAlloc_t>::deallocate(fCharAlloc, p, fBuffersSize);
         fBufStack.pop();
      }
   }
};

std::stack<void *> *RBufferStack::Get(std::size_t typeSize)
{
   thread_local RBufferStackWrapper bsw(typeSize);
   return bsw.Get();
}
} // namespace VecOps
} // namespace Internal
} // namespace ROOT
