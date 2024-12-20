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

#ifndef ROOT7_RNTuple
#define ROOT7_RNTuple

#include <Rtypes.h>

#include <cstdint>

class TCollection;
class TFile;
class TFileMergeInfo;

namespace ROOT {

class RNTuple;

namespace Experimental {
namespace Internal {
class RNTupleFileWriter;
class RPageSourceFile;

RNTuple CreateAnchor(std::uint16_t versionEpoch, std::uint16_t versionMajor, std::uint16_t versionMinor,
                     std::uint16_t versionPatch, std::uint64_t seekHeader, std::uint64_t nbytesHeader,
                     std::uint64_t lenHeader, std::uint64_t seekFooter, std::uint64_t nbytesFooter,
                     std::uint64_t lenFooter, std::uint64_t maxKeySize);

} // namespace Internal
} // namespace Experimental

// clang-format off
/**
\class ROOT::RNTuple
\ingroup NTuple
\brief Representation of an RNTuple data set in a ROOT file

The class points to the header and footer keys, which in turn have the references to the pages (via page lists).
Only the RNTuple key will be listed in the list of keys. Like TBaskets, the pages are "invisible" keys.
Byte offset references in the RNTuple header and footer reference directly the data part of page records,
skipping the TFile key part.

In the list of keys, this object appears as "ROOT::RNTuple".
It is the user-facing representation of an RNTuple data set in a ROOT file and
it provides an API entry point to an RNTuple stored in a ROOT file. Its main purpose is to
construct a page source for an RNTuple, which in turn can be used to read an RNTuple with an RDF or
an RNTupleReader.

For instance, for an RNTuple called "Events" in a ROOT file, usage can be
~~~ {.cpp}
auto f = TFile::Open("data.root");
auto ntpl = f->Get<ROOT::RNTuple>("Events");
auto reader = RNTupleReader::Open(ntpl);
~~~
*/
// clang-format on
class RNTuple final {
   friend class Experimental::Internal::RNTupleFileWriter;
   friend class Experimental::Internal::RPageSourceFile;

   friend ROOT::RNTuple ROOT::Experimental::Internal::CreateAnchor(
      std::uint16_t versionEpoch, std::uint16_t versionMajor, std::uint16_t versionMinor, std::uint16_t versionPatch,
      std::uint64_t seekHeader, std::uint64_t nbytesHeader, std::uint64_t lenHeader, std::uint64_t seekFooter,
      std::uint64_t nbytesFooter, std::uint64_t lenFooter, std::uint64_t maxKeySize);

public:
   static constexpr std::uint16_t kVersionEpoch = 1;
   static constexpr std::uint16_t kVersionMajor = 0;
   static constexpr std::uint16_t kVersionMinor = 0;
   static constexpr std::uint16_t kVersionPatch = 0;

private:
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
   /// The maximum size for a TKey payload. Payloads bigger than this size will be written as multiple blobs.
   std::uint64_t fMaxKeySize = 0;

   TFile *fFile = nullptr; ///<! The file from which the ntuple was streamed, registered in the custom streamer

public:
   RNTuple() = default;
   ~RNTuple() = default;

   std::uint16_t GetVersionEpoch() const { return fVersionEpoch; }
   std::uint16_t GetVersionMajor() const { return fVersionMajor; }
   std::uint16_t GetVersionMinor() const { return fVersionMinor; }
   std::uint16_t GetVersionPatch() const { return fVersionPatch; }

   std::uint64_t GetSeekHeader() const { return fSeekHeader; }
   std::uint64_t GetNBytesHeader() const { return fNBytesHeader; }
   std::uint64_t GetLenHeader() const { return fLenHeader; }

   std::uint64_t GetSeekFooter() const { return fSeekFooter; }
   std::uint64_t GetNBytesFooter() const { return fNBytesFooter; }
   std::uint64_t GetLenFooter() const { return fLenFooter; }
   std::uint64_t GetMaxKeySize() const { return fMaxKeySize; }

   /// RNTuple implements the hadd MergeFile interface
   /// Merge this NTuple with the input list entries
   Long64_t Merge(TCollection *input, TFileMergeInfo *mergeInfo);

   /// NOTE: if you change this version you also need to update RTFNTuple::fClassVersion in RMiniFile.cxx
   ClassDefNV(RNTuple, 2);
}; // class RNTuple

} // namespace ROOT

#endif
