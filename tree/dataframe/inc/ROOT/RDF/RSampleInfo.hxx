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

#include <ROOT/RDF/RSample.hxx>
#include <string_view>
#include <Rtypes.h>

#include <functional>
#include <stdexcept>
#include <string>

#include <tuple>

namespace ROOT {
namespace RDF {

/// This type represents a sample identifier, to be used in conjunction with RDataFrame features such as
/// \ref ROOT::RDF::RInterface< Proxied, DS_t >::DefinePerSample "DefinePerSample()" and per-sample callbacks.
///
/// When the input data comes from a TTree, the string representation of RSampleInfo (which is returned by AsString()
/// and that can be queried e.g. with Contains()) is of the form "<filename>/<treename>".
///
/// In multi-thread runs, different tasks might process different entry ranges of the same sample,
/// so RSampleInfo also provides methods to inspect which part of a sample is being taken into consideration.
class RSampleInfo {
   std::string fID;
   std::pair<ULong64_t, ULong64_t> fEntryRange;

   const ROOT::RDF::Experimental::RSample *fSample = nullptr; // non-owning

   void ThrowIfNoSample() const
   {
      if (fSample == nullptr) {
         const auto msg = "RSampleInfo: sample data was requested but no samples are available.";
         throw std::logic_error(msg);
      }
   }

public:
   RSampleInfo(std::string_view id, std::pair<ULong64_t, ULong64_t> entryRange,
               const ROOT::RDF::Experimental::RSample *sample = nullptr)
      : fID(id), fEntryRange(entryRange), fSample(sample)
   {
   }
   RSampleInfo() = default;
   RSampleInfo(const RSampleInfo &) = default;
   RSampleInfo &operator=(const RSampleInfo &) = default;
   RSampleInfo(RSampleInfo &&) = default;
   RSampleInfo &operator=(RSampleInfo &&) = default;
   ~RSampleInfo() = default;

   /// @brief Get the name of the sample as a string.
   const std::string &GetSampleName() const
   {
      ThrowIfNoSample();
      return fSample->GetSampleName();
   }

   /// @brief Get the sample id as an int.
   unsigned int GetSampleId() const
   {
      ThrowIfNoSample();
      return fSample->GetSampleId();
   }

   /// @brief Return the metadata value of type int given the key.
   int GetI(const std::string &key) const
   {
      ThrowIfNoSample();
      return fSample->GetMetaData().GetI(key);
   }

   /// @brief Return the metadata value of type double given the key.
   double GetD(const std::string &key) const
   {
      ThrowIfNoSample();
      return fSample->GetMetaData().GetD(key);
   }

   /// @brief Return the metadata value of type string given the key.
   std::string GetS(const std::string &key) const
   {
      ThrowIfNoSample();
      return fSample->GetMetaData().GetS(key);
   }

   /// @brief Check whether the sample name contains the given substring.
   bool Contains(std::string_view substr) const
   {
      // C++14 needs the conversion from std::string_view to std::string
      return fID.find(std::string(substr)) != std::string::npos;
   }

   /// @brief Check whether the sample name is empty.
   ///
   /// This is the case e.g. when using a RDataFrame with no input data, constructed as `RDataFrame(nEntries)`.
   bool Empty() const {
      return fID.empty();
   }

   /// @brief Return a string representation of the sample name.
   ///
   /// The representation is of the form "<filename>/<treename>" if the input data comes from a TTree or a TChain.
   const std::string &AsString() const
   {
      return fID;
   }

   /// @brief Return the entry range in the sample that is being taken into consideration.
   ///
   /// Multiple multi-threading tasks might process different entry ranges of the same sample.
   std::pair<ULong64_t, ULong64_t> EntryRange() const { return fEntryRange; }

   /// @brief Return the number of entries of this sample that is being taken into consideration.
   ULong64_t NEntries() const { return fEntryRange.second - fEntryRange.first; }

   bool operator==(const RSampleInfo &other) const { return fID == other.fID; }
   bool operator!=(const RSampleInfo &other) const { return !(*this == other); }
};

/// The type of a data-block callback, registered with an RDataFrame computation graph via e.g.  \ref
/// ROOT::RDF::RInterface< Proxied, DS_t >::DefinePerSample "DefinePerSample()" or by certain actions (e.g. \ref
/// ROOT::RDF::RInterface<Proxied,DataSource>::Snapshot "Snapshot()").
using SampleCallback_t = std::function<void(unsigned int, const ROOT::RDF::RSampleInfo &)>;

} // namespace RDF
} // namespace ROOT

#endif
