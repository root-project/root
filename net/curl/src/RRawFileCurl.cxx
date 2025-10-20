// @(#)root/net:$Id$
// Author: Jakob Blomer

#include "ROOT/RCurlConnection.hxx"
#include "ROOT/RError.hxx"
#include "ROOT/RRawFileCurl.hxx"

#include <utility>

ROOT::Internal::RRawFileCurl::RRawFileCurl(std::string_view url, ROptions options) : RRawFile(url, options) {}

ROOT::Internal::RRawFileCurl::~RRawFileCurl() = default;

std::unique_ptr<ROOT::Internal::RRawFile> ROOT::Internal::RRawFileCurl::Clone() const
{
   return std::make_unique<RRawFileCurl>(fUrl, fOptions);
}

void ROOT::Internal::RRawFileCurl::OpenImpl()
{
   fConnection = std::make_unique<RCurlConnection>(GetUrl());
   if (fOptions.fBlockSize == ROptions::kUseDefaultBlockSize)
      fOptions.fBlockSize = kDefaultBlockSize;
}

std::uint64_t ROOT::Internal::RRawFileCurl::GetSizeImpl()
{
   std::uint64_t size;
   auto status = fConnection->SendHeadReq(size);
   if (!status)
      throw RException(R__FAIL("cannot determine file size of " + GetUrl() + ": " + status.fStatusMsg));
   return size;
}

std::size_t ROOT::Internal::RRawFileCurl::ReadAtImpl(void *buffer, std::size_t nbytes, std::uint64_t offset)
{
   RCurlConnection::RUserRange range;
   range.fDestination = reinterpret_cast<unsigned char *>(buffer);
   range.fOffset = offset;
   range.fLength = nbytes;
   auto status = fConnection->SendRangesReq(1, &range);
   if (!status)
      throw RException(R__FAIL("cannot read from " + GetUrl() + ": " + status.fStatusMsg));
   return range.fNBytesRecv;
}

void ROOT::Internal::RRawFileCurl::ReadVImpl(RIOVec *ioVec, unsigned int nReq)
{
   if (nReq == 0)
      return;

   std::vector<RCurlConnection::RUserRange> ranges;
   ranges.reserve(nReq);

   for (unsigned int i = 0; i < nReq; ++i) {
      RCurlConnection::RUserRange range;
      range.fDestination = reinterpret_cast<unsigned char *>(ioVec[i].fBuffer);
      range.fOffset = ioVec[i].fOffset;
      range.fLength = ioVec[i].fSize;
      ranges.emplace_back(range);
   }

   auto status = fConnection->SendRangesReq(ranges.size(), &ranges[0]);
   if (!status)
      throw RException(R__FAIL("cannot read from " + GetUrl() + ": " + status.fStatusMsg));

   for (unsigned int i = 0; i < nReq; ++i) {
      ioVec[i].fOutBytes = ranges[i].fNBytesRecv;
   }
}

ROOT::Internal::RCurlConnection &ROOT::Internal::RRawFileCurl::GetConnection()
{
   EnsureOpen();
   return *fConnection;
}
