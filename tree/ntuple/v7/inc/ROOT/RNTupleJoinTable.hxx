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
   /// fields) to their respective entry numbers.
   std::unordered_map<RCombinedJoinFieldValue, std::vector<ROOT::NTupleSize_t>, RCombinedJoinFieldValueHash> fJoinTable;

   /// The page source belonging to the RNTuple for which to build the join table.
   std::unique_ptr<RPageSource> fPageSource;

   /// The fields for which the join table is built. Used to compute the hashes for each entry value.
   std::vector<std::unique_ptr<RFieldBase>> fJoinFields;

   /// Only built join tables can be queried.
   bool fIsBuilt = false;

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Create an a new RNTupleJoinTable for the RNTuple represented by the provided page source.
   ///
   /// \param[in] fieldNames The names of the join fields to use for the join table. Only integral-type fields are
   /// allowed.
   //  \param[in] pageSource The page source.
   RNTupleJoinTable(const std::vector<std::string> &fieldNames, const RPageSource &pageSource);

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
   /// \param[in] pageSource The page source.
   ///
   /// \return A pointer to the newly-created join table.
   static std::unique_ptr<RNTupleJoinTable>
   Create(const std::vector<std::string> &fieldNames, const RPageSource &pageSource);

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Build the join table.
   ///
   /// Only a built join table can be queried (with RNTupleJoinTable::GetFirstEntryNumber or
   /// RNTupleJoinTable::GetAllEntryNumbers).
   void Build();

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
   /// \brief Get the first entry number corresponding to the given join field value(s).
   ///
   /// \param[in] valuePtrs A vector of pointers to the join field values to look up.
   ///
   /// \return The first entry number that corresponds to `valuePtrs`. When no such entry exists, `kInvalidNTupleIndex`
   /// is returned.
   ///
   /// Note that in case multiple entries corresponding to the provided join field value exist, the first occurrence is
   /// returned. Use RNTupleJoinTable::GetAllEntryNumbers to get all entries.
   ROOT::NTupleSize_t GetFirstEntryNumber(const std::vector<void *> &valuePtrs) const;

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Get the entry number corresponding to the given join field value(s).
   ///
   /// \sa GetFirstEntryNumber(std::vector<void *> valuePtrs)
   template <typename... Ts>
   ROOT::NTupleSize_t GetFirstEntryNumber(Ts... values) const
   {
      if (sizeof...(Ts) != fJoinFields.size())
         throw RException(R__FAIL("number of values must match number of join fields"));

      std::vector<void *> valuePtrs;
      valuePtrs.reserve(sizeof...(Ts));
      ([&] { valuePtrs.push_back(&values); }(), ...);

      return GetFirstEntryNumber(valuePtrs);
   }

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Get all entry numbers for the given join field value(s).
   ///
   /// \param[in] valuePtrs A vector of pointers to the join field values to look up.
   ///
   /// \return The entry numbers that corresponds to `valuePtrs`. When no such entry exists, an empty vector is
   /// returned.
   const std::vector<ROOT::NTupleSize_t> *GetAllEntryNumbers(const std::vector<void *> &valuePtrs) const;

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Get all entry numbers for the given join field value(s).
   ///
   /// \sa GetAllEntryNumbers(std::vector<void *> valuePtrs)
   template <typename... Ts>
   const std::vector<ROOT::NTupleSize_t> *GetAllEntryNumbers(Ts... values) const
   {
      if (sizeof...(Ts) != fJoinFields.size())
         throw RException(R__FAIL("number of values must match number of join fields"));

      std::vector<void *> valuePtrs;
      valuePtrs.reserve(sizeof...(Ts));
      ([&] { valuePtrs.push_back(&values); }(), ...);

      return GetAllEntryNumbers(valuePtrs);
   }
};
} // namespace Internal
} // namespace Experimental
} // namespace ROOT

#endif // ROOT7_RNTupleJoinTable
