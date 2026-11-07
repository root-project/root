#ifndef ROOT_INTERNAL_IMTUTILS
#define ROOT_INTERNAL_IMTUTILS

#include <functional>
#include <memory>

namespace ROOT {
class TThreadExecutor;
}

namespace ROOT::Internal::IMTUtils {

/// \brief Abstract implementation of the oneAPI Range requirements
///
/// For the specification, see
/// https://oneapi-spec.uxlfoundation.org/specifications/oneapi/latest/elements/onetbb/source/named_requirements/algorithms/range.
///
/// The class is meant to pass implementations from other parts of ROOT to the
/// internal calls to the TBB parallel_for overload taking a Range object.
class RAbstractTBBRangeImpl {
public:
   virtual std::unique_ptr<RAbstractTBBRangeImpl> SplitWork() = 0;
   virtual bool Empty() const = 0;
   virtual bool IsDivisible() const = 0;
   RAbstractTBBRangeImpl() = default;

   // Destructor is required for dynamic polymorphism, so apply rule of five
   virtual ~RAbstractTBBRangeImpl() = default;
   RAbstractTBBRangeImpl(const RAbstractTBBRangeImpl &) = default;
   RAbstractTBBRangeImpl &operator=(const RAbstractTBBRangeImpl &) = default;
   RAbstractTBBRangeImpl(RAbstractTBBRangeImpl &&) = default;
   RAbstractTBBRangeImpl &operator=(RAbstractTBBRangeImpl &&) = default;
};

// Entry point to a parallel_for that takes a Range and Body
// https://oneapi-spec.uxlfoundation.org/specifications/oneapi/latest/elements/onetbb/source/algorithms/functions/parallel_for_func
void ParallelFor(ROOT::TThreadExecutor &pool, std::unique_ptr<RAbstractTBBRangeImpl> rangeImpl,
                 const std::function<void(const RAbstractTBBRangeImpl &)> &body);

} // namespace ROOT::Internal::IMTUtils

#endif
