/// \file RNTuple.cxx
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

#include <ROOT/RError.hxx>
#include <ROOT/RNTuple.hxx>
#include <ROOT/RPageStorage.hxx>
#include <ROOT/RPageStorageFile.hxx>

#include <TBuffer.h>
#include <TError.h>
#include <TFile.h>

#include <xxhash.h>

void ROOT::RNTuple::Streamer(TBuffer &buf)
{
   if (buf.IsReading()) {
      // Skip byte count and class version
      auto offCkData = buf.Length() + sizeof(UInt_t) + sizeof(Version_t);
      buf.ReadClassBuffer(RNTuple::Class(), this);
      std::uint64_t expectedChecksum = XXH3_64bits(buf.Buffer() + offCkData, buf.Length() - offCkData);

      std::uint64_t onDiskChecksum;
      if (static_cast<std::size_t>(buf.BufferSize()) < buf.Length() + sizeof(onDiskChecksum))
         throw RException(R__FAIL("the buffer containing RNTuple is too small to contain the checksum!"));
      buf >> onDiskChecksum;

      if (expectedChecksum != onDiskChecksum)
         throw RException(R__FAIL("checksum mismatch in RNTuple anchor"));

      R__ASSERT(buf.GetParent() && buf.GetParent()->InheritsFrom("TFile"));
      fFile = static_cast<TFile *>(buf.GetParent());
   } else {
      auto offCkData = buf.Length() + sizeof(UInt_t) + sizeof(Version_t);
      buf.WriteClassBuffer(RNTuple::Class(), this);
      std::uint64_t checksum = XXH3_64bits(buf.Buffer() + offCkData, buf.Length() - offCkData);
      buf << checksum;
   }
}

ROOT::RNTuple ROOT::Experimental::Internal::CreateAnchor(std::uint16_t versionEpoch, std::uint16_t versionMajor,
                                                         std::uint16_t versionMinor, std::uint16_t versionPatch,
                                                         std::uint64_t seekHeader, std::uint64_t nbytesHeader,
                                                         std::uint64_t lenHeader, std::uint64_t seekFooter,
                                                         std::uint64_t nbytesFooter, std::uint64_t lenFooter,
                                                         std::uint64_t maxKeySize)
{
   RNTuple ntuple;
   ntuple.fVersionEpoch = versionEpoch;
   ntuple.fVersionMajor = versionMajor;
   ntuple.fVersionMinor = versionMinor;
   ntuple.fVersionPatch = versionPatch;
   ntuple.fSeekHeader = seekHeader;
   ntuple.fNBytesHeader = nbytesHeader;
   ntuple.fLenHeader = lenHeader;
   ntuple.fSeekFooter = seekFooter;
   ntuple.fNBytesFooter = nbytesFooter;
   ntuple.fLenFooter = lenFooter;
   ntuple.fMaxKeySize = maxKeySize;
   return ntuple;
}
