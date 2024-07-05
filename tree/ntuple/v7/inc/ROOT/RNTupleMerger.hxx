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

#include "Compression.h"
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
namespace Internal {

struct RNTupleMergeOptions {
   /// If `fCompressionSettings == -1` (the default), the merger will not change the compression
   /// of any of its sources (fast merging). Otherwise, all sources will be converted to the specified
   /// compression algorithm and level.
   int fCompressionSettings = -1;
};

// clang-format off
/**
 * \class ROOT::Experimental::Internal::RNTupleMerger
 * \ingroup NTuple
 * \brief Given a set of RPageSources merge them into an RPageSink, optionally changing their compression.
 *        This can also be used to change the compression of a single RNTuple by just passing a single source.
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
   void CollectColumns(const RNTupleDescriptor &descriptor, std::vector<RColumnInfo> &columns);

   // Internal map that holds column name, type, and type id : output ID information
   std::unordered_map<std::string, DescriptorId_t> fOutputIdMap;

public:
   /// Merge a given set of sources into the destination
   void Merge(std::span<RPageSource *> sources, RPageSink &destination,
              const RNTupleMergeOptions &options = RNTupleMergeOptions());

}; // end of class RNTupleMerger

} // namespace Internal
} // namespace Experimental
} // namespace ROOT

#endif
