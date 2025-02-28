/// \file ROOT/RNTupleJoinTable.hxx
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

#ifndef ROOT7_RNTupleJoinTable
#define ROOT7_RNTupleJoinTable

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
\class ROOT::Experimental::Internal::RNTupleJoinTable
\ingroup NTuple
\brief Builds a join table on one or several fields of an RNTuple so it can be joined onto other RNTuples.
*/
// clang-format on
class RNTupleJoinTable {
public:
   using NTupleJoinValue_t = std::uint64_t;

private:
   /////////////////////////////////////////////////////////////////////////////
   /// Container for the hashes of the join fields.
   class RCombinedJoinFieldValue {
   public:
      std::vector<NTupleJoinValue_t> fFieldValues;
      RCombinedJoinFieldValue(const std::vector<NTupleJoinValue_t> &fieldValues)
      {
         fFieldValues.reserve(fieldValues.size());
         fFieldValues = fieldValues;
      }
      inline bool operator==(const RCombinedJoinFieldValue &other) const { return other.fFieldValues == fFieldValues; }
   };

   /////////////////////////////////////////////////////////////////////////////
   /// Hash combining the individual join field value hashes from RCombinedJoinFieldValue. Uses the implementation from
   /// `boost::hash_combine` (see
   /// https://www.boost.org/doc/libs/1_55_0/doc/html/hash/reference.html#boost.hash_combine).
   struct RCombinedJoinFieldValueHash {
      inline std::size_t operator()(const RCombinedJoinFieldValue &joinFieldValue) const
      {
         std::size_t combinedHash = 0;
         for (const auto &fieldVal : joinFieldValue.fFieldValues) {
            combinedHash ^= fieldVal + 0x9e3779b9 + (fieldVal << 6) + (fieldVal >> 2);
         }
         return combinedHash;
      }
   };

   /// The join table itself. Maps field values (or combinations thereof in case the join table is defined for multiple
   /// fields) to their respective entry indexes.
   std::unordered_map<RCombinedJoinFieldValue, std::vector<ROOT::NTupleSize_t>, RCombinedJoinFieldValueHash> fJoinTable;

   /// Names of the join fields used for the mapping to their respective entry indexes.
   std::vector<std::string> fJoinFieldNames;

   /// The size (in bytes) for each join field, corresponding to `fJoinFieldNames`. This information is stored to be
   /// able to properly cast incoming void pointers to the join field values in `GetEntryIndexes`.
   std::vector<std::size_t> fJoinFieldValueSizes;

   /// Only built join tables can be queried.
   bool fIsBuilt = false;

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Create an a new RNTupleJoinTable for the RNTuple represented by the provided page source.
   ///
   /// \param[in] fieldNames The names of the join fields to use for the join table. Only integral-type fields are
   /// allowed.
   RNTupleJoinTable(const std::vector<std::string> &fieldNames) : fJoinFieldNames(fieldNames) {}

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Ensure the RNTupleJoinTable has been built.
   ///
   /// \throws RException If the join table has not been built, and can therefore not be used yet.
   void EnsureBuilt() const;

public:
   RNTupleJoinTable(const RNTupleJoinTable &other) = delete;
   RNTupleJoinTable &operator=(const RNTupleJoinTable &other) = delete;
   RNTupleJoinTable(RNTupleJoinTable &&other) = delete;
   RNTupleJoinTable &operator=(RNTupleJoinTable &&other) = delete;
   ~RNTupleJoinTable() = default;

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Create an RNTupleJoinTable from an existing RNTuple.
   ///
   /// \param[in] fieldNames The names of the join fields to use for the join table. Only integral-type fields are
   /// allowed.
   ///
   /// \return A pointer to the newly-created join table.
   static std::unique_ptr<RNTupleJoinTable> Create(const std::vector<std::string> &fieldNames);

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Build the join table.
   ///
   /// \param[in] pageSource The page source of the RNTuple for which to build the join table.
   ///
   /// Only a built join table can be queried (with RNTupleJoinTable::GetEntryIndexes).
   void Build(RPageSource &pageSource);

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Get the number of entries in the join table.
   ///
   /// \return The number of entries in the join table.
   ///
   /// \note This does not have to correspond to the number of entries in the original RNTuple. If the original RNTuple
   /// contains duplicate join field values, they are counted as one.
   std::size_t GetSize() const
   {
      EnsureBuilt();
      return fJoinTable.size();
   }

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Whether the join table has been built (and therefore ready to be used).
   ///
   /// \return `true` if the join table has been built.
   ///
   /// Only built join tables can be queried.
   bool IsBuilt() const { return fIsBuilt; }

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Get all entry indexes for the given join field value(s).
   ///
   /// \param[in] valuePtrs A vector of pointers to the join field values to look up.
   ///
   /// \return The entry indexes that correspond to `valuePtrs`. An empty vector is returned when there are no matching
   /// indexes.
   std::vector<ROOT::NTupleSize_t> GetEntryIndexes(const std::vector<void *> &valuePtrs) const;
};
} // namespace Internal
} // namespace Experimental
} // namespace ROOT

#endif // ROOT7_RNTupleJoinTable
