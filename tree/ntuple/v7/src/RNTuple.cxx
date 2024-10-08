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
      UInt_t offClassBuf;
      UInt_t bcnt;
      auto classVersion = buf.ReadVersion(&offClassBuf, &bcnt);

      // Strip class version from checksum calculation
      UInt_t lenStrip = sizeof(Version_t);

      if (bcnt < lenStrip)
         throw Experimental::RException(R__FAIL("invalid anchor byte count: " + std::to_string(bcnt)));

      auto lenCkData = bcnt - lenStrip;
      // Skip byte count and class version
      auto offCkData = offClassBuf + sizeof(UInt_t) + sizeof(Version_t);
      auto expectedChecksum = XXH3_64bits(buf.Buffer() + offCkData, lenCkData);

      std::uint64_t onDiskChecksum;
      buf.ReadClassBuffer(RNTuple::Class(), this, classVersion, offClassBuf, bcnt);
      if (static_cast<std::size_t>(buf.BufferSize()) < buf.Length() + sizeof(onDiskChecksum))
         throw Experimental::RException(R__FAIL("the buffer containing RNTuple is too small to contain the checksum!"));
      buf >> onDiskChecksum;

      if (expectedChecksum != onDiskChecksum)
         throw Experimental::RException(R__FAIL("checksum mismatch in RNTuple anchor"));

      R__ASSERT(buf.GetParent() && buf.GetParent()->InheritsFrom("TFile"));
      fFile = static_cast<TFile *>(buf.GetParent());
   } else {
      auto offCkData = buf.Length() + sizeof(UInt_t) + sizeof(Version_t);
      buf.WriteClassBuffer(RNTuple::Class(), this);
      std::uint64_t checksum = XXH3_64bits(buf.Buffer() + offCkData, buf.Length() - offCkData);
      buf << checksum;
   }
}
