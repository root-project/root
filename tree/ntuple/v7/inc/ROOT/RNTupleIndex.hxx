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

#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

namespace ROOT {
namespace Experimental {
namespace Internal {
// clang-format off
/**
\class ROOT::Experimental::Internal::RNTupleIndex
\ingroup NTuple
\brief Builds an index on one or several fields of an RNTuple so it can be joined onto other RNTuples.
*/
// clang-format on
class RNTupleIndex {
private:
   /////////////////////////////////////////////////////////////////////////////
   /// Container for the hashes of the indexed fields.
   class RIndexValue {
   public:
      std::vector<std::size_t> fValueHashes;
      RIndexValue(const std::vector<std::size_t> &valueHashes) : fValueHashes(valueHashes) {}
      inline bool operator==(const RIndexValue &other) const { return other.fValueHashes == fValueHashes; }
   };

   /////////////////////////////////////////////////////////////////////////////
   /// Hash combinining the individual index value hashes from RIndexValue. Uses the implementation from
   /// `boost::hash_combine` (see
   /// https://www.boost.org/doc/libs/1_55_0/doc/html/hash/reference.html#boost.hash_combine).
   struct RIndexValueHash {
      inline std::size_t operator()(const RIndexValue &indexValue) const
      {
         std::size_t combinedHash = 0;
         for (const auto &valueHash : indexValue.fValueHashes) {
            combinedHash ^= valueHash + 0x9e3779b9 + (valueHash << 6) + (valueHash >> 2);
         }
         return combinedHash;
      }
   };

   /// The fields for which the index is built. Used to compute the hashes for each entry value.
   const std::vector<std::unique_ptr<RFieldBase>> fFields;

   /// The index itself. Maps field values (or combinations thereof in case the index is defined for multiple fields) to
   /// their respsective entry numbers.
   std::unordered_map<RIndexValue, std::vector<NTupleSize_t>, RIndexValueHash> fIndex;

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Create an RNTupleIndex for an existing RNTuple.
   ///
   /// \param[in] The fields that will make up the index.
   /// \param[in] The number of entries to index.
   ///
   /// \note The page source is assumed be attached already.
   RNTupleIndex(std::vector<std::unique_ptr<RFieldBase>> &fields, NTupleSize_t nEntries);

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Add a new entry to the index.
   ///
   /// \param[in] valuePtrs The entry values to index, according to fFields.
   /// \param[in] entry The entry number.
   void Add(const std::vector<void *> &valuePtrs, NTupleSize_t entry);

public:
   RNTupleIndex(const RNTupleIndex &other) = delete;
   RNTupleIndex &operator=(const RNTupleIndex &other) = delete;
   RNTupleIndex(RNTupleIndex &&other) = delete;
   RNTupleIndex &operator=(RNTupleIndex &&other) = delete;
   ~RNTupleIndex() = default;

   ////////////////////////////////////////////////////////////////////////////////
   /// \brief Create an RNTupleIndex from an existing RNTuple.
   ///
   /// \param[in] fieldNames The names of the fields to index.
   /// \param pageSource The page source.
   ///
   /// \return A pointer to the newly-created index.
   ///
   static std::unique_ptr<RNTupleIndex>
   Create(const std::vector<std::string_view> &fieldNames, RPageSource &pageSource);

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Get the number of elements currently indexed.
   ///
   /// \return The number of elements currently indexed.
   std::size_t GetNElements() const { return fIndex.size(); }

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Get the first entry number containing the given index value.
   ///
   /// \param[in] valuePtrs A vector of pointers to the index values to look up.
   ///
   /// \return The first entry number that corresponds to `valuePtrs`. When no such entry exists, `kInvalidNTupleIndex`
   /// is returned.
   ///
   /// Note that in case multiple entries corresponding to the provided index value exist, the first occurrence is
   /// returned. Use RNTupleIndex::GetAllEntryNumbers to get all entries.
   NTupleSize_t GetFirstEntryNumber(const std::vector<void *> &valuePtrs) const;

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Get the entry number containing the given index value.
   ///
   /// \sa GetFirstEntryNumber(std::vector<void *> valuePtrs)
   template <typename... Ts>
   NTupleSize_t GetFirstEntryNumber(Ts... values) const
   {
      // TODO(fdegeus) also check that the types match
      if (sizeof...(Ts) != fFields.size())
         throw RException(R__FAIL("number of value pointers must match number of indexed fields"));

      std::vector<void *> valuePtrs;
      valuePtrs.reserve(sizeof...(Ts));
      ([&] { valuePtrs.push_back(&values); }(), ...);

      return GetFirstEntryNumber(valuePtrs);
   }

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Get all entry numbers for the given index.
   ///
   /// \param[in] valuePtrs A vector of pointers to the index values to look up.
   ///
   /// \return The entry numbers that corresponds to `valuePtrs`. When no such entry exists, an empty vector is
   /// returned.
   const std::vector<NTupleSize_t> *GetAllEntryNumbers(const std::vector<void *> &valuePtrs) const;

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Get all entry numbers for the given index.
   ///
   /// \sa GetAllEntryNumbers(std::vector<void *> valuePtrs)
   template <typename... Ts>
   const std::vector<NTupleSize_t> *GetAllEntryNumbers(Ts... values) const
   {
      // TODO(fdegeus) also check that the types match
      if (sizeof...(Ts) != fFields.size())
         throw RException(R__FAIL("number of value pointers must match number of indexed fields"));

      std::vector<void *> valuePtrs;
      valuePtrs.reserve(sizeof...(Ts));
      ([&] { valuePtrs.push_back(&values); }(), ...);

      return GetAllEntryNumbers(valuePtrs);
   }
};
} // namespace Internal
} // namespace Experimental
} // namespace ROOT

#endif // ROOT7_RNTupleIndex
