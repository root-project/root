/// \file ROOT/RNTuple.hxx
/// \ingroup NTuple ROOT7
/// \author Jakob Blomer <jblomer@cern.ch>
/// \date 2023-09-19
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

/*************************************************************************
 * Copyright (C) 1995-2023, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT7_RNTupleAnchor
#define ROOT7_RNTupleAnchor

#include <ROOT/RNTupleOptions.hxx>

#include <Rtypes.h>

#include <cstdint>
#include <memory>

class TCollection;
class TFile;
class TFileMergeInfo;

namespace ROOT {
namespace Experimental {

class RNTupleReadOptions;
namespace Detail {
class RPageSource;
}
namespace Internal {
class RNTupleFileWriter;
}

// clang-format off
/**
\class ROOT::Experimental::RNTuple
\ingroup NTuple
\brief Representation of an RNTuple data set in a ROOT file

The class points to the header and footer keys, which in turn have the references to the pages (via page lists).
Only the RNTuple key will be listed in the list of keys. Like TBaskets, the pages are "invisible" keys.
Byte offset references in the RNTuple header and footer reference directly the data part of page records,
skipping the TFile key part.

In the list of keys, this object appears as "ROOT::Experimental::RNTuple".
It is the user-facing representation of an RNTuple data set in a ROOT file and
it provides an API entry point to an RNTuple stored in a ROOT file. Its main purpose is to
construct a page source for an RNTuple, which in turn can be used to read an RNTuple with an RDF or
an RNTupleReader.

For instance, for an RNTuple called "Events" in a ROOT file, usage can be
~~~ {.cpp}
auto f = TFile::Open("data.root");
auto ntpl = f->Get<ROOT::Experimental::RNTuple>("Events");

auto reader = RNTupleReader::Open(ntpl);
or
auto pageSource = ntpl->MakePageSource();
~~~
*/
// clang-format on
class RNTuple final {
public:
   static constexpr std::uint16_t kVersionEpoch = 0;
   static constexpr std::uint16_t kVersionMajor = 2;
   static constexpr std::uint16_t kVersionMinor = 0;
   static constexpr std::uint16_t kVersionPatch = 0;

   /// Version of the RNTuple binary format that the writer supports (see specification).
   /// Changing the epoch indicates backward-incompatible changes
   std::uint16_t fVersionEpoch = kVersionEpoch;
   /// Changing the major version indicates forward incompatible changes; such changes should correspond to a new
   /// bit in the feature flag of the RNTuple header.
   /// For the pre-release epoch 0, indicates the release candidate number
   std::uint16_t fVersionMajor = kVersionMajor;
   /// Changing the minor version indicates new optional fields added to the RNTuple meta-data
   std::uint16_t fVersionMinor = kVersionMinor;
   /// Changing the patch version indicates new backported features from newer binary format versions
   std::uint16_t fVersionPatch = kVersionPatch;
   /// The file offset of the header excluding the TKey part
   std::uint64_t fSeekHeader = 0;
   /// The size of the compressed ntuple header
   std::uint64_t fNBytesHeader = 0;
   /// The size of the uncompressed ntuple header
   std::uint64_t fLenHeader = 0;
   /// The file offset of the footer excluding the TKey part
   std::uint64_t fSeekFooter = 0;
   /// The size of the compressed ntuple footer
   std::uint64_t fNBytesFooter = 0;
   /// The size of the uncompressed ntuple footer
   std::uint64_t fLenFooter = 0;
   /// The xxhash3 checksum of the serialized other members of the struct (excluding byte count and class version).
   /// This member can only be interpreted during streaming.
   /// When adding new members to the class, this member must remain the last one.
   std::uint64_t fChecksum = 0;

   TFile *fFile = nullptr; ///<! The file from which the ntuple was streamed, registered in the custom streamer

   RNTuple() = default;
   ~RNTuple() = default;

   /// Create a page source from the RNTuple object. Requires the RNTuple object to be streamed from a file.
   /// If fFile is not set, an exception is thrown.
   std::unique_ptr<Detail::RPageSource> MakePageSource(const RNTupleReadOptions &options = RNTupleReadOptions());

   /// RNTuple implements the hadd MergeFile interface
   /// Merge this NTuple with the input list entries
   Long64_t Merge(TCollection *input, TFileMergeInfo *mergeInfo);

   ClassDefNV(RNTuple, 4);
};

} // namespace Experimental
} // namespace ROOT

#endif
