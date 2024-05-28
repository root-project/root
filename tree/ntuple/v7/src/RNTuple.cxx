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

      // Strip class version and the fChecksum member from checksum calculation
      const UInt_t lenStrip = sizeof(fChecksum) + sizeof(Version_t);
      if (bcnt < lenStrip)
         throw RException(R__FAIL("invalid anchor byte count: " + std::to_string(bcnt)));
      auto lenCkData = bcnt - lenStrip;
      // Skip byte count and class version
      auto offCkData = offClassBuf + sizeof(UInt_t) + sizeof(Version_t);
      auto checksum = XXH3_64bits(buf.Buffer() + offCkData, lenCkData);

      buf.ReadClassBuffer(RNTuple::Class(), this, classVersion, offClassBuf, bcnt);

      if (checksum != fChecksum)
         throw RException(R__FAIL("checksum mismatch in RNTuple anchor"));

      R__ASSERT(buf.GetParent() && buf.GetParent()->InheritsFrom("TFile"));
      fFile = static_cast<TFile *>(buf.GetParent());
   } else {
      auto offBcnt = buf.WriteVersion(RNTuple::Class(), kTRUE /* useBcnt */);
      auto offCkData = buf.GetCurrent() - buf.Buffer();
      buf << fVersionEpoch;
      buf << fVersionMajor;
      buf << fVersionMinor;
      buf << fVersionPatch;
      buf << fSeekHeader;
      buf << fNBytesHeader;
      buf << fLenHeader;
      buf << fSeekFooter;
      buf << fNBytesFooter;
      buf << fLenFooter;
      fChecksum = XXH3_64bits(buf.Buffer() + offCkData, buf.Length() - offCkData);
      buf << fChecksum;
      buf.SetByteCount(offBcnt, kTRUE /* packInVersion */);
   }
}
