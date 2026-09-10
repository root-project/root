/// \file
/// \warning This is part of the %ROOT 7 prototype! It will change without notice. It might trigger earthquakes.
/// Feedback is welcome!

#ifndef ROOT_RProfileFillContext
#define ROOT_RProfileFillContext

#include "RProfile.hxx"
#include "RHistEngine.hxx"
#include "RHistStats.hxx"
#include "RHistUtils.hxx"
#include "RWeight.hxx"

#include <cstddef>
#include <tuple>

namespace ROOT {
namespace Experimental {

/**
A context to concurrently fill an RProfile.

\sa RProfileConcurrentFiller

\warning This is part of the %ROOT 7 prototype! It will change without notice. It might trigger earthquakes.
Feedback is welcome!
*/
class RProfileFillContext final {
   friend class RProfileConcurrentFiller;

private:
   /// A pointer to the filled profile histogram
   RProfile *fProfile = nullptr;

   /// Local histogram statistics
   RHistStats fStats;

   /// \sa RProfileConcurrentFiller::CreateFillContent()
   explicit RProfileFillContext(RProfile &profile) : fProfile(&profile), fStats(profile.GetNDimensions() + 1)
   {
      // Propagate disabled dimensions to the local histogram statistics object.
      const auto &stats = profile.GetStats();
      for (std::size_t i = 0; i < stats.GetNDimensions(); i++) {
         if (!stats.IsEnabled(i)) {
            fStats.DisableDimension(i);
         }
      }
   }
   RProfileFillContext(const RProfileFillContext &) = delete;
   RProfileFillContext(RProfileFillContext &&) = default;
   RProfileFillContext &operator=(const RProfileFillContext &) = delete;
   RProfileFillContext &operator=(RProfileFillContext &&) = default;

public:
   ~RProfileFillContext() { Flush(); }

   /// Fill an entry into the profile histogram.
   ///
   /// If one of the arguments is outside the corresponding axis and flow bins are disabled, the entry will be silently
   /// discarded.
   ///
   /// Throws an exception if the number of arguments does not match the axis configuration, or if an argument cannot be
   /// converted for the axis type at run-time.
   ///
   /// \param[in] args the arguments for each axis
   /// \param[in] v the additional argument
   /// \sa RProfile::Fill(const std::tuple<A...> &args, const V &value)
   template <typename... A, typename V>
   void Fill(const std::tuple<A...> &args, const V &value)
   {
      RProfile::RValueWrapper wrapper(value);
      fProfile->fEngine.FillAtomic(args, wrapper);
      // Avoid a second conversion of value, which we already did in wrapper.
      fStats.Fill(Internal::AppendReference(args, wrapper.fValue));
   }

   /// Fill an entry into the profile histogram with a weight.
   ///
   /// If one of the arguments is outside the corresponding axis and flow bins are disabled, the entry will be silently
   /// discarded.
   ///
   /// Throws an exception if the number of arguments does not match the axis configuration, or if an argument cannot be
   /// converted for the axis type at run-time.
   ///
   /// \param[in] args the arguments for each axis
   /// \param[in] v the additional argument
   /// \param[in] weight the weight for this entry
   /// \sa RProfile::Fill(const std::tuple<A...> &args, const V &value, RWeight weight)
   template <typename... A, typename V>
   void Fill(const std::tuple<A...> &args, const V &value, RWeight weight)
   {
      RProfile::RValueWeightWrapper wrapper(value, weight.fValue);
      fProfile->fEngine.FillAtomic(args, wrapper);
      // Avoid a second conversion of value, which we already did in wrapper.
      fStats.Fill(Internal::AppendReference(args, wrapper.fValue), weight);
   }

   /// Fill an entry into the profile histogram.
   ///
   /// For weighted filling, pass an RWeight as the last argument.
   ///
   /// If one of the arguments is outside the corresponding axis and flow bins are disabled, the entry will be silently
   /// discarded.
   ///
   /// Throws an exception if the number of arguments does not match the axis configuration, or if an argument cannot be
   /// converted for the axis type at run-time.
   ///
   /// \param[in] args the arguments for each axis
   /// \sa RProfile::Fill(const A &...args)
   template <typename... A>
   void Fill(const A &...args)
   {
      static_assert(sizeof...(A) >= 2, "need at least two arguments to Fill");
      if constexpr (sizeof...(A) >= 2) {
         auto t = std::forward_as_tuple(args...);
         if constexpr (std::is_same_v<typename Internal::LastType<A...>::type, RWeight>) {
            static constexpr std::size_t N = sizeof...(A) - 2;
            if (N != fProfile->GetNDimensions()) {
               throw std::invalid_argument("invalid number of arguments to Fill");
            }
            RWeight weight = std::get<N + 1>(t);
            RProfile::RValueWeightWrapper wrapper(std::get<N>(t), weight.fValue);
            fProfile->fEngine.FillAtomicImpl<N>(t, wrapper);
         } else {
            static constexpr std::size_t N = sizeof...(A) - 1;
            if (N != fProfile->GetNDimensions()) {
               throw std::invalid_argument("invalid number of arguments to Fill");
            }
            RProfile::RValueWrapper wrapper(std::get<N>(t));
            fProfile->fEngine.FillAtomicImpl<N>(t, wrapper);
         }
         fStats.Fill(args...);
      }
   }

   /// Flush locally accumulated entries to the profile histogram.
   void Flush()
   {
      fProfile->fStats.AddAtomic(fStats);
      fStats.Clear();
   }
};

} // namespace Experimental
} // namespace ROOT

#endif
