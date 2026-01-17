// @(#)root/net:$Id$
// Author: Jakob Blomer

/*************************************************************************
 * Copyright (C) 1995-2025, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TCurlFile.h"
#include "TError.h"

#include <cstdint>
#include <cstring>
#include <vector>

TCurlFile::TCurlFile(const char *url, Option_t *opt)
   : TFile(url, strstr(opt, "_WITHOUT_GLOBALREGISTRATION") != nullptr ? "WEB_WITHOUT_GLOBALREGISTRATION" : "WEB"),
     fConnection(new ROOT::Internal::RCurlConnection(url))
{
   TFile::Init(kFALSE);
   fOffset = 0;
   fD = -2; // so TFile::IsOpen() will return true when in ~TFile
}

Long64_t TCurlFile::GetSize() const
{
   std::uint64_t size;
   auto status = fConnection->SendHeadReq(size);
   if (!status)
      return -1;
   return size;
}

void TCurlFile::Seek(Long64_t offset, ERelativeTo pos)
{
   switch (pos) {
   case kBeg: fOffset = offset + fArchiveOffset; break;
   case kCur: fOffset += offset; break;
   case kEnd:
      // this option is not used currently in the ROOT code
      if (fArchiveOffset)
         Error("Seek", "seeking from end in archive is not (yet) supported");
      fOffset = fEND - offset; // is fEND really EOF or logical EOF?
      break;
   }
}

Bool_t TCurlFile::ReadBuffer(char *buf, Int_t len)
{
   ROOT::Internal::RCurlConnection::RUserRange range;
   range.fDestination = reinterpret_cast<unsigned char *>(buf);
   range.fOffset = fOffset;
   range.fLength = len;
   auto status = fConnection->SendRangesReq(1, &range);
   if (!status) {
      Error("TCurlFile", "can not read data: %s", status.fStatusMsg.c_str());
      return kTRUE;
   }

   fOffset += range.fNBytesRecv;
   return kFALSE;
}

Bool_t TCurlFile::ReadBuffer(char *buf, Long64_t pos, Int_t len)
{
   ROOT::Internal::RCurlConnection::RUserRange range;
   range.fDestination = reinterpret_cast<unsigned char *>(buf);
   range.fOffset = pos;
   range.fLength = len;
   auto status = fConnection->SendRangesReq(1, &range);
   if (!status) {
      Error("TCurlFile", "can not read data: %s", status.fStatusMsg.c_str());
      return kTRUE;
   }
   return kFALSE;
}

Bool_t TCurlFile::ReadBuffers(char *buf, Long64_t *pos, Int_t *len, Int_t nbuf)
{
   if (nbuf == 0)
      return kFALSE;

   std::vector<ROOT::Internal::RCurlConnection::RUserRange> ranges;
   ranges.reserve(nbuf);
   std::size_t bufPos = 0;
   for (Int_t i = 0; i < nbuf; ++i) {
      ROOT::Internal::RCurlConnection::RUserRange r;
      r.fDestination = reinterpret_cast<unsigned char *>(&buf[bufPos]);
      r.fOffset = pos[i];
      r.fLength = len[i];
      ranges.emplace_back(r);
      bufPos += len[i];
   }

   auto status = fConnection->SendRangesReq(nbuf, &ranges[0]);
   if (!status) {
      Error("TCurlFile", "can not read data: %s", status.fStatusMsg.c_str());
      return kTRUE;
   }
   return kFALSE;
}
