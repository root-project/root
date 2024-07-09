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
#include <Compression.h>

#include <memory>
#include <string>
#include <vector>
#include <unordered_map>

namespace ROOT {
namespace Experimental {
namespace Internal {

enum class RNTupleMergingMode {
   /// The merger will discard all columns that aren't present in the prototype model (i.e. the model of the first
   /// source)
   kFilter,
   /// The merger will update the output model to include all columns from all sources. Entries corresponding to columns
   /// that are not present in a source will be set to the default value of the type.
   kUnion
};

struct RNTupleMergeOptions {
   /// If `fCompressionSettings == kUnknownCompressionSettings` (the default), the merger will not change the
   /// compression of any of its sources (fast merging). Otherwise, all sources will be converted to the specified
   /// compression algorithm and level.
   int fCompressionSettings = kUnknownCompressionSettings;
   /// Determines how the merging treats sources with different models (\see RNTupleMergingMode).
   RNTupleMergingMode fMergingMode = RNTupleMergingMode::kFilter;
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
   // Internal map that holds column name, type, and type id : output ID information
   std::unordered_map<std::string, DescriptorId_t> fOutputIdMap;

   // Struct to hold column information
   struct RColumnInfo {
      /// The qualified field name to which the column belongs, followed by the column index, type and version
      std::string fColumnNameTypeAndVersion;
      DescriptorId_t fColumnInputId;
      DescriptorId_t fColumnOutputId;

      RColumnInfo(const std::string &name, const std::string &typeAndVersion, const DescriptorId_t &inputId,
                  const DescriptorId_t &outputId)
         : fColumnNameTypeAndVersion(name + "." + typeAndVersion), fColumnInputId(inputId), fColumnOutputId(outputId)
      {
      }
   };

   /// Validate the columns against the internal map that is built from the first source
   /// This is where we assign the output ids for the remaining sources
   void ValidateColumns(std::vector<RColumnInfo> &columns) const;

   /// Recursively add columns from a given field.
   /// The columns are added in such a way that all already-seen columns (i.e. the ones whose name and version
   /// were already in `fOutputIdMap`) are at the end of `columns` and all new ones are at the start of it.
   /// Old and new columns preserve their relative order, so this input:
   ///
   ///    [old0, old1, new0, old2, new1, new2]
   ///
   /// will be mapped to this output:
   ///
   ///    [new0, new1, new2, old0, old1, old2]
   ///
   /// Returns the number of new columns.
   size_t AddColumnsFromField(std::vector<RColumnInfo> &columns, const RNTupleDescriptor &desc,
                              const RFieldDescriptor &fieldDesc, const std::string &prefix = "") const;

   /// Recursively collect all the columns for all the fields rooted at field zero
   /// Returns the number of new columns added.
   size_t CollectColumns(const RNTupleDescriptor &descriptor, RNTupleMergingMode mergingMode,
                         std::vector<RColumnInfo> &columns);

   /// Adds the new columns `newCols` to the destination's model `model`.
   void ExtendOutputModel(RNTupleModel &model, std::span<RColumnInfo> newCols, int nDstEntries,
                          const RNTupleDescriptor &descriptor, RPageSink &destination) const;

public:
   /// Merge a given set of sources into the destination
   void Merge(std::span<RPageSource *> sources, RPageSink &destination,
              const RNTupleMergeOptions &options = RNTupleMergeOptions());

}; // end of class RNTupleMerger

} // namespace Internal
} // namespace Experimental
} // namespace ROOT

#endif
