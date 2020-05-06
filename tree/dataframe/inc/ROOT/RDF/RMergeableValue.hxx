// Author: Vincenzo Eduardo Padulano, Enrico Guiraud 05/2020

/*************************************************************************
 * Copyright (C) 1995-2020, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_RDF_RMERGEABLEVALUE
#define ROOT_RDF_RMERGEABLEVALUE

#include <memory>
#include <stdexcept>
#include <algorithm> // std::min, std::max

#include "RtypesCore.h"
#include "TList.h" // RMergeableFill::Merge

namespace ROOT {
namespace Detail {
namespace RDF {

// Fwd declarations for RMergeableValue
template <typename T>
class RMergeableValue;

template <typename T, typename... Ts>
std::unique_ptr<RMergeableValue<T>> MergeValues(std::unique_ptr<RMergeableValue<T>> OutputMergeable,
                                                std::unique_ptr<RMergeableValue<Ts>>... InputMergeables);

template <typename T, typename... Ts>
void MergeValues(RMergeableValue<T> &OutputMergeable, const RMergeableValue<Ts> &... InputMergeables);

class RMergeableValueBase {
public:
   virtual ~RMergeableValueBase() = default;
   RMergeableValueBase() = default;
};

template <typename T>
class RMergeableValue : public RMergeableValueBase {
   // Friend function declarations
   template <typename T1, typename... Ts>
   friend std::unique_ptr<RMergeableValue<T1>> MergeValues(std::unique_ptr<RMergeableValue<T1>> OutputMergeable,
                                                           std::unique_ptr<RMergeableValue<Ts>>... InputMergeables);
   template <typename T1, typename... Ts>
   friend void MergeValues(RMergeableValue<T1> &OutputMergeable, const RMergeableValue<Ts> &... InputMergeables);

   virtual void Merge(const RMergeableValue<T> &) = 0;

protected:
   T fValue;

public:
   RMergeableValue(const T &value) : fValue{value} {}
   RMergeableValue() = default;

   const T &GetValue() const { return fValue; }
};

class RMergeableCount final : public RMergeableValue<ULong64_t> {

   void Merge(const RMergeableValue<ULong64_t> &other) final
   {
      try {
         const auto &othercast = dynamic_cast<const RMergeableCount &>(other);
         this->fValue += othercast.fValue;
      } catch (const std::bad_cast &) {
         throw std::invalid_argument("Results from different actions cannot be merged together.");
      }
   }

public:
   RMergeableCount(ULong64_t value) : RMergeableValue<ULong64_t>(value) {}
   RMergeableCount() = default;
   RMergeableCount(RMergeableCount &&) = default;
   RMergeableCount(const RMergeableCount &) = delete;
};

template <typename T>
class RMergeableFill final : public RMergeableValue<T> {
   void Merge(const RMergeableValue<T> &other) final
   {
      try {
         const auto &othercast = dynamic_cast<const RMergeableFill<T> &>(other);
         TList l;                                     // The `Merge` method accepts a TList
         l.Add(const_cast<T *>(&(othercast.fValue))); // Ugly but needed because of the signature of TList::Add
         this->fValue.Merge(&l); // if `T == TH1D` Eventually calls TH1::ExtendAxis that creates new instances of TH1D
      } catch (const std::bad_cast &) {
         throw std::invalid_argument("Results from different actions cannot be merged together.");
      }
   }

public:
   RMergeableFill(const T &value) : RMergeableValue<T>(value) {}
   RMergeableFill() = default;
   RMergeableFill(RMergeableFill &&) = default;
   RMergeableFill(const RMergeableFill &) = delete;
};

template <typename T>
class RMergeableMax final : public RMergeableValue<T> {

   void Merge(const RMergeableValue<T> &other) final
   {
      try {
         const auto &othercast = dynamic_cast<const RMergeableMax<T> &>(other);
         this->fValue = std::max(this->fValue, othercast.fValue);
      } catch (const std::bad_cast &) {
         throw std::invalid_argument("Results from different actions cannot be merged together.");
      }
   }

public:
   RMergeableMax(const T &value) : RMergeableValue<T>(value) {}
   RMergeableMax() = default;
   RMergeableMax(RMergeableMax &&) = default;
   RMergeableMax(const RMergeableMax &) = delete;
};

class RMergeableMean final : public RMergeableValue<Double_t> {
   ULong64_t fCounts;

   void Merge(const RMergeableValue<Double_t> &other) final
   {
      try {
         const auto &othercast = dynamic_cast<const RMergeableMean &>(other);
         const auto &othervalue = othercast.fValue;
         const auto &othercounts = othercast.fCounts;
         const auto num = this->fValue * fCounts + othervalue * othercounts;
         const auto denum = static_cast<Double_t>(fCounts + othercounts);
         this->fValue = num / denum;
         fCounts += othercounts;
      } catch (const std::bad_cast &) {
         throw std::invalid_argument("Results from different actions cannot be merged together.");
      }
   }

public:
   /////////////////////////////////////////////////////////////////////////////
   /// \brief Constructor that initializes data members.
   /// \param[in] value The action result.
   /// \param[in] counts The number of entries used to compute that result.
   RMergeableMean(Double_t value, ULong64_t counts) : RMergeableValue<Double_t>(value), fCounts{counts} {}
   /**
      Default constructor. Needed to allow serialization of ROOT objects. See
      [TBufferFile::WriteObjectClass]
      (classTBufferFile.html#a209078a4cb58373b627390790bf0c9c1)
   */
   RMergeableMean() = default;
   RMergeableMean(RMergeableMean &&) = default;
   RMergeableMean(const RMergeableMean &) = delete;
};

template <typename T>
class RMergeableMin final : public RMergeableValue<T> {

   void Merge(const RMergeableValue<T> &other) final
   {
      try {
         const auto &othercast = dynamic_cast<const RMergeableMin<T> &>(other);
         this->fValue = std::min(this->fValue, othercast.fValue);
      } catch (const std::bad_cast &) {
         throw std::invalid_argument("Results from different actions cannot be merged together.");
      }
   }

public:
   RMergeableMin(const T &value) : RMergeableValue<T>(value) {}
   RMergeableMin() = default;
   RMergeableMin(RMergeableMin &&) = default;
   RMergeableMin(const RMergeableMin &) = delete;
};

class RMergeableStdDev final : public RMergeableValue<Double_t> {
   ULong64_t fCounts;
   Double_t fMean;

   void Merge(const RMergeableValue<Double_t> &other) final
   {
      try {
         const auto &othercast = dynamic_cast<const RMergeableStdDev &>(other);
         const auto &othercounts = othercast.fCounts;
         const auto &othermean = othercast.fMean;

         const auto thisvariance = std::pow(this->fValue, 2);
         const auto othervariance = std::pow(othercast.fValue, 2);

         const auto delta = othermean - fMean;

         const auto m_a = thisvariance * (fCounts - 1);
         const auto m_b = othervariance * (othercounts - 1);

         const auto sumcounts = static_cast<Double_t>(fCounts + othercounts);

         const auto M2 = m_a + m_b + std::pow(delta, 2) * fCounts * othercounts / sumcounts;

         const auto meannum = fMean * fCounts + othermean * othercounts;

         this->fValue = std::sqrt(M2 / (sumcounts - 1));
         fMean = meannum / sumcounts;
         fCounts += othercounts;
      } catch (const std::bad_cast &) {
         throw std::invalid_argument("Results from different actions cannot be merged together.");
      }
   }

public:
   /////////////////////////////////////////////////////////////////////////////
   /// \brief Constructor that initializes data members.
   /// \param[in] value The action result.
   /// \param[in] counts The number of entries of the set.
   /// \param[in] mean The average of the set.
   RMergeableStdDev(Double_t value, ULong64_t counts, Double_t mean)
      : RMergeableValue<Double_t>(value), fCounts{counts}, fMean{mean}
   {
   }
   RMergeableStdDev() = default;
   RMergeableStdDev(RMergeableStdDev &&) = default;
   RMergeableStdDev(const RMergeableStdDev &) = delete;
};

template <typename T>
class RMergeableSum final : public RMergeableValue<T> {

   void Merge(const RMergeableValue<T> &other) final
   {
      try {
         const auto &othercast = dynamic_cast<const RMergeableSum<T> &>(other);
         this->fValue += othercast.fValue;
      } catch (const std::bad_cast &) {
         throw std::invalid_argument("Results from different actions cannot be merged together.");
      }
   }

public:
   RMergeableSum(const T &value) : RMergeableValue<T>(value) {}
   RMergeableSum() = default;
   RMergeableSum(RMergeableSum &&) = default;
   RMergeableSum(const RMergeableSum &) = delete;
};

// What follows mimics C++17 std::conjunction without using recursive template instantiations.
// Used in `MergeValues` to check that all the mergeables hold values of the same type.
template <bool...>
struct bool_pack {
};
template <class... Ts>
using conjunction = std::is_same<bool_pack<true, Ts::value...>, bool_pack<Ts::value..., true>>;

template <typename T, typename... Ts>
std::unique_ptr<RMergeableValue<T>> MergeValues(std::unique_ptr<RMergeableValue<T>> OutputMergeable,
                                                std::unique_ptr<RMergeableValue<Ts>>... InputMergeables)
{
   // Check all mergeables have the same template type
   static_assert(conjunction<std::is_same<Ts, T>...>::value, "Values must all be of the same type.");

   // Using dummy array initialization inspired by https://stackoverflow.com/a/25683817
   using expander = int[];
   // Cast to void to suppress unused-value warning in Clang
   (void)expander{0, (OutputMergeable->Merge(*InputMergeables), 0)...};

   return OutputMergeable;
}

template <typename T, typename... Ts>
void MergeValues(RMergeableValue<T> &OutputMergeable, const RMergeableValue<Ts> &... InputMergeables)
{
   // Check all mergeables are of the same type
   static_assert(conjunction<std::is_same<Ts, T>...>::value, "Values must all be of the same type.");

   // Using dummy array initialization inspired by https://stackoverflow.com/a/25683817
   using expander = int[];
   // Cast to void to suppress unused-value warning in Clang
   (void)expander{0, (OutputMergeable.Merge(InputMergeables), 0)...};
}
} // namespace RDF
} // namespace Detail
} // namespace ROOT

#endif // ROOT_RDF_RMERGEABLEVALUE