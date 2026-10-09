/// \file
/// \warning This is part of the %ROOT 7 prototype! It will change without notice. It might trigger earthquakes.
/// Feedback is welcome!

#ifndef ROOT_RProfileConcurrentFiller
#define ROOT_RProfileConcurrentFiller

#include "RProfile.hxx"
#include "RProfileFillContext.hxx"

#include <exception>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <vector>

namespace ROOT {
namespace Experimental {

/**
A histogram filler to concurrently fill an RProfile.

\code
auto profile = std::make_shared<ROOT::Experimental::RProfile>(10, std::make_pair(5, 15));
{
   ROOT::Experimental::RProfileConcurrentFiller filler(profile);
   auto context = filler.CreateFillContext();
   context->Fill(8.5, 23.0);
}
// profile->GetBinContent(ROOT::Experimental::RBinIndex(3)) will return the filled bin content
\endcode

\warning This is part of the %ROOT 7 prototype! It will change without notice. It might trigger earthquakes.
Feedback is welcome!
*/
class RProfileConcurrentFiller final {
   /// A pointer to the filled profile histogram
   std::shared_ptr<RProfile> fProfile;

   /// Mutex to protect access to the list of fill contexts (not for filling itself!)
   std::mutex fMutex;
   /// The list of fill contexts, for checks during destruction
   std::vector<std::weak_ptr<RProfileFillContext>> fFillContexts;

public:
   /// Create a filler object.
   ///
   /// \param[in] hist a pointer to the histogram
   explicit RProfileConcurrentFiller(std::shared_ptr<RProfile> profile) : fProfile(profile)
   {
      if (!profile) {
         throw std::invalid_argument("profile must not be nullptr");
      }
   }

   RProfileConcurrentFiller(const RProfileConcurrentFiller &) = delete;
   RProfileConcurrentFiller(RProfileConcurrentFiller &&) = delete;
   RProfileConcurrentFiller &operator=(const RProfileConcurrentFiller &) = delete;
   RProfileConcurrentFiller &operator=(RProfileConcurrentFiller &&) = delete;

   ~RProfileConcurrentFiller()
   {
      for (const auto &context : fFillContexts) {
         if (!context.expired()) {
            // According to C++ Core Guideline C.36 "A destructor must not fail" and (C.37) "If a destructor tries to
            // exit with an exception, it’s a bad design error and the program had better terminate".
            std::terminate(); // GCOVR_EXCL_LINE
         }
      }
   }

   const std::shared_ptr<RProfile> &GetProfile() const { return fProfile; }

   /// Create a new context for concurrent filling.
   std::shared_ptr<RProfileFillContext> CreateFillContext()
   {
      // Cannot use std::make_shared because the constructor of RProfileFillContext is private. Also it would mean that
      // the (direct) memory of all contexts stays around until the vector of weak_ptr's is cleared.
      std::shared_ptr<RProfileFillContext> context(new RProfileFillContext(*fProfile));

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
