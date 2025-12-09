/// \file
/// \warning This is part of the %ROOT 7 prototype! It will change without notice. It might trigger earthquakes.
/// Feedback is welcome!

#ifndef ROOT_RHistFillContext
#define ROOT_RHistFillContext

#include "RHist.hxx"
#include "RHistEngine.hxx"
#include "RHistStats.hxx"
#include "RWeight.hxx"

#include <tuple>

namespace ROOT {
namespace Experimental {

// forward declaration for friend declaration
template <typename BinContentType>
class RHistConcurrentFiller;

/**
A context to concurrently fill an RHist.

\sa RHistConcurrentFiller

\warning This is part of the %ROOT 7 prototype! It will change without notice. It might trigger earthquakes.
Feedback is welcome!
*/
template <typename BinContentType>
class RHistFillContext final {
   friend class RHistConcurrentFiller<BinContentType>;

private:
   /// A pointer to the filled histogram
   RHist<BinContentType> *fHist = nullptr;

   /// Local histogram statistics
   RHistStats fStats;

   /// \sa RHistConcurrentFiller::CreateFillContent()
   explicit RHistFillContext(RHist<BinContentType> &hist) : fHist(&hist), fStats(hist.GetNDimensions()) {}
   RHistFillContext(const RHistFillContext &) = delete;
   RHistFillContext(RHistFillContext &&) = default;
   RHistFillContext &operator=(const RHistFillContext &) = delete;
   RHistFillContext &operator=(RHistFillContext &&) = default;

public:
   ~RHistFillContext() { Flush(); }

   /// Fill an entry into the histogram.
   ///
   /// If one of the arguments is outside the corresponding axis and flow bins are disabled, the entry will be silently
   /// discarded.
   ///
   /// Throws an exception if the number of arguments does not match the axis configuration, or if an argument cannot be
   /// converted for the axis type at run-time.
   ///
   /// \param[in] args the arguments for each axis
   /// \sa RHist::Fill(const std::tuple<A...> &args)
   template <typename... A>
   void Fill(const std::tuple<A...> &args)
   {
      fHist->fEngine.FillAtomic(args);
      fStats.Fill(args);
   }

   /// Fill an entry into the histogram with a weight.
   ///
   /// This overload is not available for integral bin content types (see \ref RHistEngine::SupportsWeightedFilling).
   ///
   /// If one of the arguments is outside the corresponding axis and flow bins are disabled, the entry will be silently
   /// discarded.
   ///
   /// Throws an exception if the number of arguments does not match the axis configuration, or if an argument cannot be
   /// converted for the axis type at run-time.
   ///
   /// \param[in] args the arguments for each axis
   /// \param[in] weight the weight for this entry
   /// \sa RHist::Fill(const std::tuple<A...> &args, RWeight weight)
   template <typename... A>
   void Fill(const std::tuple<A...> &args, RWeight weight)
   {
      fHist->fEngine.FillAtomic(args, weight);
      fStats.Fill(args, weight);
   }

   /// Fill an entry into the histogram.
   ///
   /// For weighted filling, pass an RWeight as the last argument. This is not available for integral bin content types
   /// (see \ref RHistEngine::SupportsWeightedFilling).
   ///
   /// If one of the arguments is outside the corresponding axis and flow bins are disabled, the entry will be silently
   /// discarded.
   ///
   /// Throws an exception if the number of arguments does not match the axis configuration, or if an argument cannot be
   /// converted for the axis type at run-time.
   ///
   /// \param[in] args the arguments for each axis
   /// \sa RHist::Fill(const A &...args)
   template <typename... A>
   void Fill(const A &...args)
   {
      fHist->fEngine.FillAtomic(args...);
      fStats.Fill(args...);
   }

   /// Flush locally accumulated entries to the histogram.
   void Flush()
   {
      fHist->fStats.AddAtomic(fStats);
      fStats.Clear();
   }
};

} // namespace Experimental
} // namespace ROOT

#endif
