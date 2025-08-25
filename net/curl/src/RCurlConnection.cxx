// @(#)root/net:$Id$
// Author: Jakob Blomer

#include "ROOT/RCurlConnection.hxx"
#include "ROOT/RError.hxx"

#include <string>

#include <curl/curl.h>

namespace {

std::string GetCurlErrorString(CURLcode code)
{
   return std::string(curl_easy_strerror(code)) + " (" + std::to_string(code) + ")";
}

} // anonymous namespace

ROOT::Internal::RCurlConnection::RCurlConnection(const std::string & /* url */)
{
   static const auto initCode = curl_global_init(CURL_GLOBAL_DEFAULT);
   if (initCode != CURLE_OK) {
      throw RException(R__FAIL("cannot initialize curl library: " + GetCurlErrorString(initCode)));
   }

   fHandle = curl_easy_init();
   if (!fHandle) {
      throw RException(R__FAIL("cannot initialize curl handle"));
   }
}

ROOT::Internal::RCurlConnection::~RCurlConnection()
{
   if (fHandle)
      curl_easy_cleanup(fHandle);
}
