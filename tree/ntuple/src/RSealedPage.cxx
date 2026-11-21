/// \file RSealedPage.cxx
/// \author Jakob Blomer <jblomer@cern.ch>
/// \date 2026-11-21

/*************************************************************************
 * Copyright (C) 1995-2026, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <ROOT/RError.hxx>
#include <ROOT/RNTupleSerialize.hxx>
#include <ROOT/RSealedPage.hxx>

void ROOT::Internal::RSealedPage::ChecksumIfEnabled()
{
   if (!fHasChecksum)
      return;

   auto charBuf = reinterpret_cast<const unsigned char *>(fBuffer);
   auto checksumBuf = const_cast<unsigned char *>(charBuf) + GetDataSize();
   std::uint64_t xxhash3;
   RNTupleSerializer::SerializeXxHash3(charBuf, GetDataSize(), xxhash3, checksumBuf);
}

ROOT::RResult<void> ROOT::Internal::RSealedPage::VerifyChecksumIfEnabled() const
{
   if (!fHasChecksum)
      return RResult<void>::Success();

   auto success = RNTupleSerializer::VerifyXxHash3(reinterpret_cast<const unsigned char *>(fBuffer), GetDataSize());
   if (!success)
      return R__FAIL("page checksum verification failed, data corruption detected");
   return RResult<void>::Success();
}

ROOT::RResult<std::uint64_t> ROOT::Internal::RSealedPage::GetChecksum() const
{
   if (!fHasChecksum)
      return R__FAIL("invalid attempt to extract non-existing page checksum");

   assert(fBufferSize >= kNBytesPageChecksum);
   std::uint64_t checksum;
   RNTupleSerializer::DeserializeUInt64(
      reinterpret_cast<const unsigned char *>(fBuffer) + fBufferSize - kNBytesPageChecksum, checksum);
   return checksum;
}
