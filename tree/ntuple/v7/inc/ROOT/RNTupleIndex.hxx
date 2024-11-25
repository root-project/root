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

#include <memory>
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
public:
   using NTupleIndexValue_t = std::uint64_t;

private:
   /////////////////////////////////////////////////////////////////////////////
   /// Container for the hashes of the indexed fields.
   class RIndexValue {
   public:
      std::vector<NTupleIndexValue_t> fFieldValues;
      RIndexValue(const std::vector<NTupleIndexValue_t> &fieldValues)
      {
         fFieldValues.reserve(fieldValues.size());
         fFieldValues = fieldValues;
      }
      inline bool operator==(const RIndexValue &other) const { return other.fFieldValues == fFieldValues; }
   };

   /////////////////////////////////////////////////////////////////////////////
   /// Hash combinining the individual index value hashes from RIndexValue. Uses the implementation from
   /// `boost::hash_combine` (see
   /// https://www.boost.org/doc/libs/1_55_0/doc/html/hash/reference.html#boost.hash_combine).
   struct RIndexValueHash {
      inline std::size_t operator()(const RIndexValue &indexValue) const
      {
         std::size_t combinedHash = 0;
         for (const auto &fieldVal : indexValue.fFieldValues) {
            combinedHash ^= fieldVal + 0x9e3779b9 + (fieldVal << 6) + (fieldVal >> 2);
         }
         return combinedHash;
      }
   };

   /// The index itself. Maps field values (or combinations thereof in case the index is defined for multiple fields) to
   /// their respsective entry numbers.
   std::unordered_map<RIndexValue, std::vector<NTupleSize_t>, RIndexValueHash> fIndex;

   /// The page source belonging to the RNTuple for which to build the index.
   std::unique_ptr<RPageSource> fPageSource;

   /// The fields for which the index is built. Used to compute the hashes for each entry value.
   std::vector<std::unique_ptr<RFieldBase>> fIndexFields;

   /// Only built indexes can be queried.
   bool fIsBuilt = false;

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Create an a new RNTupleIndex for the RNTuple represented by the provided page source.
   ///
   /// \param[in] fieldNames The names of the fields to index. Only integral-type fields can be specified as index
   /// fields.
   /// \param[in] pageSource The page source.
   RNTupleIndex(const std::vector<std::string> &fieldNames, const RPageSource &pageSource);

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Ensure the RNTupleIndex has been built.
   ///
   /// \throws RException If the index has not been built, and can therefore not be used yet.
   void EnsureBuilt() const;

public:
   RNTupleIndex(const RNTupleIndex &other) = delete;
   RNTupleIndex &operator=(const RNTupleIndex &other) = delete;
   RNTupleIndex(RNTupleIndex &&other) = delete;
   RNTupleIndex &operator=(RNTupleIndex &&other) = delete;
   ~RNTupleIndex() = default;

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Create an RNTupleIndex from an existing RNTuple.
   ///
   /// \param[in] fieldNames The names of the fields to index. Only integral-type fields can be specified as index
   /// fields.
   /// \param[in] pageSource The page source.
   /// \param[in] deferBuild When set to `true`, an empty index will be created. A call to RNTupleIndex::Build is
   /// required before the index can actually be used.
   ///
   /// \return A pointer to the newly-created index.
   static std::unique_ptr<RNTupleIndex>
   Create(const std::vector<std::string> &fieldNames, const RPageSource &pageSource, bool deferBuild = false);

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Build the index.
   ///
   /// Only a built index can be queried (with RNTupleIndex::GetFirstEntryNumber or RNTupleIndex::GetAllEntryNumbers).
   void Build();

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Get the number of indexed values.
   ///
   /// \return The number of indexed values.
   ///
   /// \note This does not have to correspond to the number of entries in the original RNTuple. If the original RNTuple
   /// contains duplicate index values, they are counted as one.
   std::size_t GetSize() const
   {
      EnsureBuilt();
      return fIndex.size();
   }

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Whether the index has been built (and therefore ready to be used).
   ///
   /// \return `true` if the index has been built.
   ///
   /// Only built indexes can be queried.
   bool IsBuilt() const { return fIsBuilt; }

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
      if (sizeof...(Ts) != fIndexFields.size())
         throw RException(R__FAIL("number of values must match number of indexed fields"));

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
      if (sizeof...(Ts) != fIndexFields.size())
         throw RException(R__FAIL("number of values must match number of indexed fields"));

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
