// Author: Enrico Guiraud, 2021

/*************************************************************************
 * Copyright (C) 1995-2021, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_RDF_RSAMPLEINFO
#define ROOT_RDF_RSAMPLEINFO

#include <ROOT/RStringView.hxx>
#include <Rtypes.h>

#include <functional>
#include <stdexcept>
#include <string>

namespace ROOT {
namespace RDF {

/// This type represents a sample identifier, to be used in conjunction with RDataFrame features such as
/// DefinePerSample() and per-sample callbacks.
///
/// When the input data comes from a TTree, the string representation of RSampleInfo (which is returned by AsString()
/// and that can be queried e.g. with Contains()) is of the form "<filename>/<treename>".
///
/// In multi-thread runs different tasks might process different entry ranges of the same sample,
/// so RSampleInfo also provides methods to inspect which part of a sample is being taken into consideration.
class RSampleInfo {
   // Currently backed by a simple string, might change in the future as we get usage experience.
   std::string fID;
   std::pair<ULong64_t, ULong64_t> fEntryRange;

public:
   explicit RSampleInfo(std::string_view id, std::pair<ULong64_t, ULong64_t> entryRange)
      : fID(id), fEntryRange(entryRange)
   {
   }
   RSampleInfo() = default;
   RSampleInfo(const RSampleInfo &) = default;
   RSampleInfo &operator=(const RSampleInfo &) = default;
   RSampleInfo(RSampleInfo &&) = default;
   RSampleInfo &operator=(RSampleInfo &&) = default;
   ~RSampleInfo() = default;

   /// Check whether the sample name contains the given substring.
   bool Contains(std::string_view substr) const
   {
      // C++14 needs the conversion from std::string_view to std::string
      return fID.find(std::string(substr)) != std::string::npos;
   }

   /// Check whether the sample name is empty.
   ///
   /// This is the case e.g. when using a RDataFrame with no input data, constructed as `RDataFrame(nEntries)`.
   bool Empty() const {
      return fID.empty();
   }

   /// Return a string representation of the sample name.
   ///
   /// The representation is of the form "<filename>/<treename>" if the input data comes from a TTree or a TChain.
   const std::string &AsString() const
   {
      return fID;
   }

   /// Return the entry range in this sample that is being taken into consideration.
   ///
   /// Multiple multi-threading tasks might process different entry ranges of the same sample.
   std::pair<ULong64_t, ULong64_t> EntryRange() const { return fEntryRange; }

   /// Return the number of entries of this sample that is being taken into consideration.
   ULong64_t NEntries() const { return fEntryRange.second - fEntryRange.first; }

   bool operator==(const RSampleInfo &other) const { return fID == other.fID; }
   bool operator!=(const RSampleInfo &other) const { return !(*this == other); }
};

/// The type of a data-block callback, registered with a RDataFrame computation graph via e.g.
/// DefinePerSample() or by certain actions (e.g. Snapshot()).
using SampleCallback_t = std::function<void(unsigned int, const ROOT::RDF::RSampleInfo &)>;

} // namespace RDF
} // namespace ROOT

#endif
