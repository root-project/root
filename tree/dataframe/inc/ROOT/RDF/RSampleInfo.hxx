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

/// This type represents a data-block identifier, to be used in conjunction with RDataFrame features such as
/// DefinePerSample() and data-block callbacks.
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

   bool Contains(std::string_view substr) const
   {
      // C++14 needs the conversion from std::string_view to std::string
      return fID.find(std::string(substr)) != std::string::npos;
   }

   bool Empty() const {
      return fID.empty();
   }

   const std::string &AsString() const
   {
      return fID;
   }

   std::pair<ULong64_t, ULong64_t> EntryRange() const { return fEntryRange; }

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
