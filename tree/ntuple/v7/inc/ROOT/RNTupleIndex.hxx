/// \file ROOT/RNTupleIndex.hxx
/// \ingroup NTuple ROOT7
/// \author Florine de Geus <florine.de.geus@cern.ch>
/// \date 2024-04-02
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

/*************************************************************************
 * Copyright (C) 1995-2024, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT7_RNTupleIndex
#define ROOT7_RNTupleIndex

#include <ROOT/RField.hxx>
#include <ROOT/RNTupleUtil.hxx>

#include <unordered_map>
#include <memory>
#include <set>
#include <string>
#include <vector>

namespace ROOT {
namespace Experimental {
namespace Internal {
/////////////////////////////////////////////////////////////////////////////
/// Container for the combined hash of the indexed fields. Uses the implementation from `boost::hash_combine` (see
/// https://www.boost.org/doc/libs/1_55_0/doc/html/hash/reference.html#boost.hash_combine).
struct RIndexValue {
   std::size_t fValue = 0;

   void operator+=(std::size_t other) { fValue ^= other + 0x9e3779b9 + (fValue << 6) + (fValue >> 2); }
   inline bool operator==(const RIndexValue &other) const { return other.fValue == fValue; }
   inline size_t operator()(const ROOT::Experimental::Internal::RIndexValue &val) const { return val.fValue; }
};

// clang-format off
/**
\class ROOT::Experimental::Internal::RNTupleIndex
\ingroup NTuple
\brief Build an index for an RNTuple so it can be joined onto other RNTuples.
*/
// clang-format on
class RNTupleIndex {
   friend std::unique_ptr<RNTupleIndex>
   CreateRNTupleIndex(const std::vector<std::string_view> &fieldNames, RPageSource &pageSource);

private:
   const std::vector<std::unique_ptr<RFieldBase>> fFields;
   std::unordered_map<RIndexValue, std::vector<NTupleSize_t>, RIndexValue> fIndex;

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Create an RNTupleIndex for an existing RNTuple.
   ///
   /// \param[in] The fields that will make up the index.
   /// \param[in] The page source of the RNTuple to build the index for.
   ///
   /// \note The page source is assumed be attached already.
   RNTupleIndex(std::vector<std::unique_ptr<RFieldBase>> &fields, RPageSource &pageSource);

public:
   RNTupleIndex(const RNTupleIndex &other) = delete;
   RNTupleIndex &operator=(const RNTupleIndex &other) = delete;
   RNTupleIndex(RNTupleIndex &&other) = delete;
   RNTupleIndex &operator=(RNTupleIndex &&other) = delete;

   std::size_t GetNElems() const { return fIndex.size(); }

   void Add(const std::vector<void *> &valuePtrs, NTupleSize_t entry);

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Get the entry number containing the given index value.
   ///
   /// \param[in] value The indexed value
   /// \return The entry number, containing the specified index value. When no such entry exists, return
   /// `kInvalidNTupleIndex`
   ///
   /// Note that in case multiple entries corresponding to the provided index value exist, the first occurrence is
   /// returned. Use RNTupleIndex::GetEntryIndices to get all entries.
   NTupleSize_t GetEntryIndex(const std::vector<void *> &valuePtrs) const;

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Get the entry number containing the given index value.
   ///
   /// \sa GetEntryIndex(std::vector<void *> valuePtrs)
   template <typename... Ts>
   NTupleSize_t GetEntryIndex(Ts... values) const
   {
      if (sizeof...(Ts) != fFields.size())
         throw RException(R__FAIL("number of value pointers must match number of indexed fields"));

      std::vector<void *> valuePtrs;
      valuePtrs.reserve(sizeof...(Ts));
      ([&] { valuePtrs.push_back(&values); }(), ...);

      return GetEntryIndex(valuePtrs);
   }

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Get all entry numbers for the given index.
   ///
   /// \param[in] value The indexed value
   /// \return The entry numbers containing the specified index value. When no entries exists, return an empty vector.
   std::vector<NTupleSize_t> GetEntryIndices(const std::vector<void *> &valuePtrs) const;

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Get all entry numbers for the given index.
   ///
   /// \sa GetEntryIndices(std::vector<void *> valuePtrs)
   template <typename... Ts>
   std::vector<NTupleSize_t> GetEntryIndices(Ts... values) const
   {
      if (sizeof...(Ts) != fFields.size())
         throw RException(R__FAIL("number of value pointers must match number of indexed fields"));

      std::vector<void *> valuePtrs;
      valuePtrs.reserve(sizeof...(Ts));
      ([&] { valuePtrs.push_back(&values); }(), ...);

      return GetEntryIndices(valuePtrs);
   }
};

////////////////////////////////////////////////////////////////////////////////
/// \brief Create an RNTupleIndex from an existing RNTuple.
///
/// \param[in] fieldNames The names of the fields to index.
/// \param pageSource The page source.
///
/// \return A pointer to the newly-created index.
///
std::unique_ptr<RNTupleIndex>
CreateRNTupleIndex(const std::vector<std::string_view> &fieldNames, RPageSource &pageSource);

} // namespace Internal
} // namespace Experimental
} // namespace ROOT

#endif // ROOT7_RNTupleIndex
