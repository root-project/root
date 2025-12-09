/// \file
/// \warning This is part of the %ROOT 7 prototype! It will change without notice. It might trigger earthquakes.
/// Feedback is welcome!

#ifndef ROOT_RHistConcurrentFiller
#define ROOT_RHistConcurrentFiller

#include "RHist.hxx"
#include "RHistFillContext.hxx"

#include <exception>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <vector>

namespace ROOT {
namespace Experimental {

/**
A histogram filler to concurrently fill an RHist.

\code
auto hist = std::make_shared<ROOT::Experimental::RHist<int>>(10, std::make_pair(5, 15));
{
   ROOT::Experimental::RHistConcurrentFiller filler(hist);
   auto context = filler.CreateFillContext();
   context.Fill(8.5);
}
// hist->GetBinContent(ROOT::Experimental::RBinIndex(3)) will return 1
\endcode

\warning This is part of the %ROOT 7 prototype! It will change without notice. It might trigger earthquakes.
Feedback is welcome!
*/
template <typename BinContentType>
class RHistConcurrentFiller final {
   /// A pointer to the filled histogram
   std::shared_ptr<RHist<BinContentType>> fHist;

   /// Mutex to protect access to the list of fill contexts (not for filling itself!)
   std::mutex fMutex;
   /// The list of fill contexts, for checks during destruction
   std::vector<std::weak_ptr<RHistFillContext<BinContentType>>> fFillContexts;

public:
   /// Create a filler object.
   ///
   /// \param[in] hist a pointer to the histogram
   explicit RHistConcurrentFiller(std::shared_ptr<RHist<BinContentType>> hist) : fHist(hist)
   {
      if (!hist) {
         throw std::invalid_argument("hist must not be nullptr");
      }
   }

   RHistConcurrentFiller(const RHistConcurrentFiller &) = delete;
   RHistConcurrentFiller(RHistConcurrentFiller &&) = delete;
   RHistConcurrentFiller &operator=(const RHistConcurrentFiller &) = delete;
   RHistConcurrentFiller &operator=(RHistConcurrentFiller &&) = delete;

   ~RHistConcurrentFiller()
   {
      for (const auto &context : fFillContexts) {
         if (!context.expired()) {
            // According to C++ Core Guideline C.36 "A destructor must not fail" and (C.37) "If a destructor tries to
            // exit with an exception, it’s a bad design error and the program had better terminate".
            std::terminate(); // GCOVR_EXCL_LINE
         }
      }
   }

   const std::shared_ptr<RHist<BinContentType>> &GetHist() const { return fHist; }

   /// Create a new context for concurrent filling.
   std::shared_ptr<RHistFillContext<BinContentType>> CreateFillContext()
   {
      // Cannot use std::make_shared because the constructor of RHistFillContext is private. Also it would mean that the
      // (direct) memory of all contexts stays around until the vector of weak_ptr's is cleared.
      std::shared_ptr<RHistFillContext<BinContentType>> context(new RHistFillContext<BinContentType>(*fHist));

      {
         std::lock_guard g(fMutex);
         fFillContexts.push_back(context);
      }

      return context;
   }
};

} // namespace Experimental
} // namespace ROOT

#endif
