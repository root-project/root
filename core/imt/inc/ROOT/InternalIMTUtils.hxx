#ifndef ROOT_INTERNAL_IMTUTILS
#define ROOT_INTERNAL_IMTUTILS

#include <memory>

namespace ROOT::Internal::IMTUtils {
class RParallelSplitFileProcessor {
public:
   virtual std::unique_ptr<RParallelSplitFileProcessor> SplitWork() = 0;
   virtual bool Empty() const = 0;
   virtual bool IsDivisible() const = 0;
   RParallelSplitFileProcessor() = default;

   // Rule of five
   virtual ~RParallelSplitFileProcessor() = default;
   RParallelSplitFileProcessor(const RParallelSplitFileProcessor &) = delete;
   RParallelSplitFileProcessor &operator=(const RParallelSplitFileProcessor &) = delete;
   RParallelSplitFileProcessor(RParallelSplitFileProcessor &&) = delete;
   RParallelSplitFileProcessor &operator=(RParallelSplitFileProcessor &&) = delete;
};
} // namespace ROOT::Internal::IMTUtils

#endif