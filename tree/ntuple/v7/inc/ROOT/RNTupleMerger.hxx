/// \file ROOT/RNTupleMerger.hxx
/// \ingroup NTuple ROOT7
/// \author Jakob Blomer <jblomer@cern.ch> & Max Orok <maxwellorok@gmail.com>
/// \date 2020-07-08
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

/*************************************************************************
 * Copyright (C) 1995-2020, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT7_RNTupleMerger
#define ROOT7_RNTupleMerger

#include <ROOT/RError.hxx>
#include <ROOT/RNTupleDescriptor.hxx>
#include <ROOT/RNTupleUtil.hxx>
#include <ROOT/RPageStorage.hxx>

#include <memory>
#include <string>
#include <vector>
#include <unordered_map>

namespace ROOT {
namespace Experimental {

// clang-format off
/**
\class ROOT::Experimental::RFieldMerger
\ingroup NTuple
\brief Two-way merge between NTuple fields
*/
// clang-format on
class RFieldMerger {
private:
   /// The merged field descriptor
   RFieldDescriptor fMergedField = RFieldDescriptor();

public:
   static RResult<RFieldMerger> Merge(const RFieldDescriptor &lhs, const RFieldDescriptor &rhs);
};

// clang-format off
/**
 * \class ROOT::Experimental::RNTupleMerger
 * \ingroup NTuple
 * \brief Given a set of RPageSources merge them into an RPageSink
 */
// clang-format on
class RNTupleMerger {

private:
   // Struct to hold column information
   struct RColumnInfo {
      std::string fColumnName;
      std::string fColumnTypeAndVersion;
      DescriptorId_t fColumnInputId;
      DescriptorId_t fColumnOutputId;

      RColumnInfo(const std::string &name, const std::string &typeAndVersion, const DescriptorId_t &inputId,
                  const DescriptorId_t &outputId)
         : fColumnName(name), fColumnTypeAndVersion(typeAndVersion), fColumnInputId(inputId), fColumnOutputId(outputId)
      {
      }
   };

   /// Build the internal column id map from the first source
   /// This is where we assign the output ids for the first source
   void BuildColumnIdMap(std::vector<RColumnInfo> &columns)
   {
      for (auto &column : columns) {
         column.fColumnOutputId = fOutputIdMap.size();
         fOutputIdMap[column.fColumnName + "." + column.fColumnTypeAndVersion] = column.fColumnOutputId;
      }
   }

   /// Validate the columns against the internal map that is built from the first source
   /// This is where we assign the output ids for the remaining sources
   void ValidateColumns(std::vector<RColumnInfo> &columns)
   {
      // First ensure that we have the same number of columns
      if (fOutputIdMap.size() != columns.size()) {
         throw RException(R__FAIL("Columns between sources do NOT match"));
      }
      // Then ensure that we have the same names of columns and assign the ids
      for (auto &column : columns) {
         try {
            column.fColumnOutputId = fOutputIdMap.at(column.fColumnName + "." + column.fColumnTypeAndVersion);
         } catch (const std::out_of_range &) {
            throw RException(R__FAIL("Column NOT found in the first source w/ name " + column.fColumnName +
                                     " type and version " + column.fColumnTypeAndVersion));
         }
      }
   }

   /// Recursively add columns from a given field
   void AddColumnsFromField(std::vector<RColumnInfo> &columns, const RNTupleDescriptor &desc,
                            const RFieldDescriptor &fieldDesc, const std::string &suffix = "")
   {
      for (const auto &field : desc.GetFieldIterable(fieldDesc)) {
         std::string name = suffix + field.GetFieldName() + ".";
         const std::string typeAndVersion = field.GetTypeName() + "." + std::to_string(field.GetTypeVersion());
         for (const auto &column : desc.GetColumnIterable(field)) {
            columns.emplace_back(name + std::to_string(column.GetIndex()), typeAndVersion, column.GetPhysicalId(),
                                 kInvalidDescriptorId);
         }
         AddColumnsFromField(columns, desc, field, name);
      }
   }

   /// Recursively collect all the columns for all the fields rooted at field zero
   std::vector<RColumnInfo> CollectColumns(const Detail::RPageSource *source, bool firstSource)
   {
      auto desc = source->GetSharedDescriptorGuard();
      std::vector<RColumnInfo> columns;
      // Here we recursively find the columns and fill the RColumnInfo vector
      AddColumnsFromField(columns, desc.GetRef(), desc->GetFieldZero());
      // Then we either build the internal map (first source) or validate the columns against it (remaning sources)
      // In either case, we also assign the output ids here
      if (firstSource) {
         BuildColumnIdMap(columns);
      } else {
         ValidateColumns(columns);
      }
      return columns;
   }

   // Internal map that holds column name, type, and type id : output ID information
   std::unordered_map<std::string, DescriptorId_t> fOutputIdMap;

public:
   /// Merge a given set of sources into the destination
   void Merge(std::span<Detail::RPageSource *> sources, Detail::RPageSink &destination);

}; // end of class RNTupleMerger

} // namespace Experimental
} // namespace ROOT

#endif
