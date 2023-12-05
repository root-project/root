/// \file ROOT/RNTupleMerger.hxx
/// \ingroup NTuple ROOT7
/// \author Jakob Blomer <jblomer@cern.ch>, Max Orok <maxwellorok@gmail.com>, Alaettin Serhan Mete <amete@anl.gov>
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
      std::string fColumnName; ///< The qualified field name to which the column belongs, followed by the column index
      std::string fColumnTypeAndVersion; ///< "<type>.<version>" of the field to which the column belongs
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
   void BuildColumnIdMap(std::vector<RColumnInfo> &columns);

   /// Validate the columns against the internal map that is built from the first source
   /// This is where we assign the output ids for the remaining sources
   void ValidateColumns(std::vector<RColumnInfo> &columns);

   /// Recursively add columns from a given field
   void AddColumnsFromField(std::vector<RColumnInfo> &columns, const RNTupleDescriptor &desc,
                            const RFieldDescriptor &fieldDesc, const std::string &prefix = "");

   /// Recursively collect all the columns for all the fields rooted at field zero
   std::vector<RColumnInfo> CollectColumns(const Detail::RPageSource &source, bool firstSource);

   // Internal map that holds column name, type, and type id : output ID information
   std::unordered_map<std::string, DescriptorId_t> fOutputIdMap;

public:
   /// Merge a given set of sources into the destination
   void Merge(std::span<Detail::RPageSource *> sources, Detail::RPageSink &destination);

}; // end of class RNTupleMerger

} // namespace Experimental
} // namespace ROOT

#endif
