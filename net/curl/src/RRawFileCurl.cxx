// @(#)root/net:$Id$
// Author: Jakob Blomer

#include "ROOT/RCurlConnection.hxx"
#include "ROOT/RRawFileCurl.hxx"

ROOT::Internal::RRawFileCurl::RRawFileCurl(std::string_view url, ROptions options)
   : RRawFile(url, options), fCurlConnection(new RCurlConnection(url))
{
}

ROOT::Internal::RRawFileCurl::~RRawFileCurl() {}

std::unique_ptr<ROOT::Internal::RRawFile> ROOT::Internal::RRawFileCurl::Clone() const
{
   return std::make_unique<RRawFileCurl>(fUrl, fOptions);
}

std::uint64_t ROOT::Internal::RRawFileCurl::GetSizeImpl()
{
   return 0;
}

void ROOT::Internal::RRawFileCurl::OpenImpl() {}

size_t ROOT::Internal::RRawFileCurl::ReadAtImpl(void * /* buffer */, size_t /* nbytes */, std::uint64_t /* offset */)
{
   return 0;
}
