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

void ROOT::Experimental::RNTuple::Streamer(TBuffer &buf)
{
   if (buf.IsReading()) {
      UInt_t offClassBuf;
      UInt_t bcnt;
      auto classVersion = buf.ReadVersion(&offClassBuf, &bcnt);
      if (classVersion < 4)
         throw RException(R__FAIL("unsupported RNTuple pre-release"));

      // Strip class version from checksum calculation
      UInt_t lenStrip = sizeof(Version_t);
      // TEMP(version4): In version 4 checksum is embedded in the on disk representation,
      // so we need to strip that as well from the byte count.
      // Support for version 4 will be dropped before the class moves out of experimental.
      lenStrip += (classVersion == 4) * sizeof(std::uint64_t);

      if (bcnt < lenStrip)
         throw RException(R__FAIL("invalid anchor byte count: " + std::to_string(bcnt)));

      auto lenCkData = bcnt - lenStrip;
      // Skip byte count and class version
      auto offCkData = offClassBuf + sizeof(UInt_t) + sizeof(Version_t);
      auto expectedChecksum = XXH3_64bits(buf.Buffer() + offCkData, lenCkData);

      std::uint64_t onDiskChecksum;
      if (classVersion == 4) {
         // TEMP(version4): Version 5 of the anchor breaks backward compat, but we still want to support version 4
         // for a while. Support for version 4, as well as this code, will be removed before the RNTuple stabilization.
         // For version 4 we need to manually read all the known members as we cannot rely on ReadClassBuffer.
         constexpr std::size_t expectedBytes = 66;
         if (bcnt != expectedBytes)
            throw RException(R__FAIL("byte count mismatch in RNTuple anchor v4: expected=" +
                                     std::to_string(expectedBytes) + ", got=" + std::to_string(bcnt)));
         buf >> fVersionEpoch;
         buf >> fVersionMajor;
         buf >> fVersionMinor;
         buf >> fVersionPatch;
         buf >> fSeekHeader;
         buf >> fNBytesHeader;
         buf >> fLenHeader;
         buf >> fSeekFooter;
         buf >> fNBytesFooter;
         buf >> fLenFooter;
         buf >> onDiskChecksum;
      } else {
         // Rewind the version bytes, as ReadClassBuffer needs to read the version again.
         buf.SetBufferOffset(offClassBuf);
         buf.ReadClassBuffer(RNTuple::Class(), this);
         if (static_cast<std::size_t>(buf.BufferSize()) < buf.Length() + sizeof(onDiskChecksum))
            throw RException(R__FAIL("the buffer containing RNTuple is too small to contain the checksum!"));
         buf >> onDiskChecksum;
      }

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


ROOT::Experimental::RNTuple ROOT::Experimental::Internal::CreateAnchor(
   std::uint16_t versionEpoch, std::uint16_t versionMajor, std::uint16_t versionMinor, std::uint16_t versionPatch,
   std::uint64_t seekHeader, std::uint64_t nbytesHeader, std::uint64_t lenHeader, std::uint64_t seekFooter,
   std::uint64_t nbytesFooter, std::uint64_t lenFooter, std::uint64_t maxKeySize)
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
