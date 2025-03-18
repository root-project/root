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
   using JoinValue_t = std::uint64_t;
   using PartitionKey_t = std::uint64_t;
   static constexpr PartitionKey_t kDefaultPartitionKey = PartitionKey_t(-1);

private:
   // clang-format off
   /**
   \class ROOT::Experimental::Internal::RNTupleJoinTable::REntryMapping
   \ingroup NTuple
   \brief Provides a mapping from one or several join field values to an entry index.
   */
   // clang-format on
   class REntryMapping {
   private:
      //////////////////////////////////////////////////////////////////////////
      /// Container for the combined hashes of join field values.
      struct RCombinedJoinFieldValue {
         std::vector<JoinValue_t> fJoinFieldValues;

         RCombinedJoinFieldValue(const std::vector<JoinValue_t> &joinFieldValues) : fJoinFieldValues(joinFieldValues) {}

         inline bool operator==(const RCombinedJoinFieldValue &other) const
         {
            return other.fJoinFieldValues == fJoinFieldValues;
         }
      };

      //////////////////////////////////////////////////////////////////////////
      /// Hash combining the individual join field value hashes from RCombinedJoinFieldValue. Uses the implementation
      /// from `boost::hash_combine` (see
      /// https://www.boost.org/doc/libs/1_55_0/doc/html/hash/reference.html#boost.hash_combine).
      struct RCombinedJoinFieldValueHash {
         inline std::size_t operator()(const RCombinedJoinFieldValue &joinFieldValue) const
         {
            std::size_t combinedHash = 0;
            for (const auto &fieldVal : joinFieldValue.fJoinFieldValues) {
               combinedHash ^= fieldVal + 0x9e3779b9 + (fieldVal << 6) + (fieldVal >> 2);
            }
            return combinedHash;
         }
      };

      /// The mapping itself. Maps field values (or combinations thereof in case the join key is composed of multiple
      /// fields) to their respective entry numbers.
      std::unordered_map<RCombinedJoinFieldValue, std::vector<ROOT::NTupleSize_t>, RCombinedJoinFieldValueHash>
         fMapping;

      /// Names of the join fields used for the mapping to their respective entry indexes.
      std::vector<std::string> fJoinFieldNames;

      /// The size (in bytes) for each join field, corresponding to `fJoinFieldNames`. This information is stored to be
      /// able to properly cast incoming void pointers to the join field values in `GetEntryIndexes`.
      std::vector<std::size_t> fJoinFieldValueSizes;

   public:
      //////////////////////////////////////////////////////////////////////////
      /// \brief Get the entry indexes for this entry mapping.
      const std::vector<ROOT::NTupleSize_t> *GetEntryIndexes(std::vector<void *> valuePtrs) const;

      //////////////////////////////////////////////////////////////////////////
      /// \brief Create a new entry mapping.
      ///
      /// \param[in] pageSource The page source of the RNTuple with the entries to map.
      /// \param[in] joinFieldNames Names of the join fields to use in the mapping.
      REntryMapping(RPageSource &pageSource, const std::vector<std::string> &joinFieldNames);
   };
   /// Names of the join fields used for the mapping to their respective entry indexes.
   std::vector<std::string> fJoinFieldNames;

   /// Partitions of one or multiple entry mappings.
   std::unordered_map<PartitionKey_t, std::vector<std::unique_ptr<REntryMapping>>> fPartitions;

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Create an a new RNTupleJoinTable for the RNTuple represented by the provided page source.
   ///
   /// \param[in] joinFieldNames The names of the join fields to use for the join table. Only integral-type fields are
   /// allowed.
   RNTupleJoinTable(const std::vector<std::string> &joinFieldNames) : fJoinFieldNames(joinFieldNames) {}

public:
   RNTupleJoinTable(const RNTupleJoinTable &other) = delete;
   RNTupleJoinTable &operator=(const RNTupleJoinTable &other) = delete;
   RNTupleJoinTable(RNTupleJoinTable &&other) = delete;
   RNTupleJoinTable &operator=(RNTupleJoinTable &&other) = delete;
   ~RNTupleJoinTable() = default;

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Create an RNTupleJoinTable from an existing RNTuple.
   ///
   /// \param[in] joinFieldNames The names of the join fields to use for the join table. Only integral-type fields are
   /// allowed.
   ///
   /// \return A pointer to the newly-created join table.
   static std::unique_ptr<RNTupleJoinTable> Create(const std::vector<std::string> &joinFieldNames);

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Add an entry mapping to the join table.
   ///
   ///
   /// \param[in] pageSource The page source of the RNTuple with the entries to map.
   /// \param[in] partitionKey Which partition to add the mapping to. If not provided, it will be added to the default
   /// partition.
   ///
   /// \return A reference to the updated join table.
   RNTupleJoinTable &Add(RPageSource &pageSource, PartitionKey_t partitionKey = kDefaultPartitionKey);

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Get all entry indexes for the given join field value(s) within a partition.
   ///
   /// \param[in] valuePtrs A vector of pointers to the join field values to look up.
   /// \param[in] partitionKey The partition key to use for the lookup. If not provided, it will use the default
   /// partition key.
   ///
   /// \return The entry numbers that correspond to `valuePtrs`. When there are no corresponding entries, an empty
   /// vector is returned.
   std::vector<ROOT::NTupleSize_t>
   GetEntryIndexes(const std::vector<void *> &valuePtrs, PartitionKey_t partitionKey = kDefaultPartitionKey) const;

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Get all entry indexes for the given join field value(s) for a specific set of partitions.
   ///
   /// \param[in] valuePtrs A vector of pointers to the join field values to look up.
   /// \param[in] partitionKeys The partition keys to use for the lookup.
   ///
   /// \return The entry numbers that correspond to `valuePtrs`, grouped by partition. When there are no corresponding
   /// entries, an empty map is returned.
   std::unordered_map<PartitionKey_t, std::vector<ROOT::NTupleSize_t>>
   GetPartitionedEntryIndexes(const std::vector<void *> &valuePtrs,
                              const std::vector<PartitionKey_t> &partitionKeys) const;

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Get all entry indexes for the given join field value(s) for all partitions.
   ///
   /// \param[in] valuePtrs A vector of pointers to the join field values to look up.
   ///
   /// \return The entry numbers that correspond to `valuePtrs`, grouped by partition. When there are no corresponding
   /// entries, an empty map is returned.
   std::unordered_map<PartitionKey_t, std::vector<ROOT::NTupleSize_t>>
   GetPartitionedEntryIndexes(const std::vector<void *> &valuePtrs) const;
};
} // namespace Internal
} // namespace Experimental
} // namespace ROOT

#endif // ROOT7_RNTupleJoinTable
