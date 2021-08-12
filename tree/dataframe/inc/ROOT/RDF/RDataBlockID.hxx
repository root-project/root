// Author: Enrico Guiraud, 2021

/*************************************************************************
 * Copyright (C) 1995-2021, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_RDF_RDATABLOCKID
#define ROOT_RDF_RDATABLOCKID

#include <ROOT/RStringView.hxx>

#include <functional>
#include <stdexcept>
#include <string>

namespace ROOT {
namespace RDF {

/// This type represents a data-block identifier, to be used in conjunction with RDataFrame features such as
/// DefinePerSample() and data-block callbacks.
class RDataBlockID {
   // Currently backed by a simple string, might change in the future as we get usage experience.
   std::string fID;

public:
   explicit RDataBlockID(std::string_view id) : fID(id) {}
   RDataBlockID() = default;
   RDataBlockID(const RDataBlockID &) = default;
   RDataBlockID &operator=(const RDataBlockID &) = default;
   RDataBlockID(RDataBlockID &&) = default;
   RDataBlockID &operator=(RDataBlockID &&) = default;
   ~RDataBlockID() = default;

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

   bool operator==(const RDataBlockID &other) const { return fID == other.fID; }
   bool operator!=(const RDataBlockID &other) const { return !(*this == other); }
};

/// The type of a data-block callback, registered with a RDataFrame computation graph via e.g.
/// DefinePerSample() or by certain actions (e.g. Snapshot()).
using DataBlockCallback_t = std::function<void(unsigned int, const ROOT::RDF::RDataBlockID &)>;

} // namespace RDF
} // namespace ROOT

#endif
