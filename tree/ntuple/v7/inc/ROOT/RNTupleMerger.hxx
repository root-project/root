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
   // Struct to hold column and field descriptors
   struct RColumnInfo {
      const DescriptorId_t fColumnInputId;
      const DescriptorId_t fColumnOutputId;

      RColumnInfo(const DescriptorId_t &inputId, const DescriptorId_t &outputId)
         : fColumnInputId(inputId), fColumnOutputId(outputId)
      {
      }
   };

   /// Recursively add columns from a given filed
   void
   AddColumnsFromField(std::vector<RColumnInfo> &vec, const RNTupleDescriptor &desc, const RFieldDescriptor &fieldDesc)
   {
      for (const auto &field : desc.GetFieldIterable(fieldDesc)) {
         for (const auto &column : desc.GetColumnIterable(field)) {
            const std::string name = field.GetFieldName() + "." + std::to_string(column.GetIndex());
            if (!fNameToOutputIdMap.count(name)) { // contains as of C++20
               fNameToOutputIdMap[name] = fNameToOutputIdMap.size();
            }
            vec.emplace_back(column.GetPhysicalId(), fNameToOutputIdMap.at(name));
         }
         AddColumnsFromField(vec, desc, field);
      }
   }

   /// Recursively collect all the columns for all the fields rooted at field zero
   std::vector<RColumnInfo> CollectColumns(const std::unique_ptr<Detail::RPageSource> &source)
   {
      auto desc = source->GetSharedDescriptorGuard();
      std::vector<RColumnInfo> columns;
      AddColumnsFromField(columns, desc.GetRef(), desc->GetFieldZero());
      return columns;
   }

   // Internal map that holds column name : output ID information
   std::unordered_map<std::string, DescriptorId_t> fNameToOutputIdMap;

public:
   /// Merge a given set of sources into the destination
   void
   Merge(std::vector<std::unique_ptr<Detail::RPageSource>> &sources, std::unique_ptr<Detail::RPageSink> &destination);

}; // end of class RNTupleMerger

} // namespace Experimental
} // namespace ROOT

#endif
