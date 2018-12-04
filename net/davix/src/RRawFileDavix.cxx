// @(#)root/io:$Id$
// Author: Jakob Blomer

/*************************************************************************
 * Copyright (C) 1995-2018, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "ROOT/RRawFileDavix.hxx"

#include <stdexcept>

#include <davix.hpp>
#include <sys/stat.h>

namespace ROOT {
namespace Detail {
namespace Internal {

struct RDavixFileDes {
   RDavixFileDes() : fd(nullptr), pos(&ctx) { }
   RDavixFileDes(const RDavixFileDes&) = delete;
   RDavixFileDes& operator=(const RDavixFileDes&) = delete;

   DAVIX_FD *fd;
   Davix::Context ctx;
   Davix::DavPosix pos;
};

} // namespace Internal
} // namespace Detail
} // namespace ROOT



size_t ROOT::Detail::RRawFileDavix::DoReadAt(void *buffer, size_t nbytes, std::uint64_t offset)
{
   if (!IsOpen()) Open();

   Davix::DavixError *err = nullptr;
   auto retval = fFileDes->pos.pread(fFileDes->fd, buffer, nbytes, offset, &err);
   if (retval < 0) {
      throw std::runtime_error("Cannot read from '" + fUrl + "', error: " + err->getErrMsg());
   }
   return static_cast<size_t>(retval);
}


bool ROOT::Detail::RRawFileDavix::IsOpen() {
   return fFileDes->fd != nullptr;
}


std::uint64_t ROOT::Detail::RRawFileDavix::DoGetSize()
{
   if (!IsOpen()) Open();

   struct stat buf;
   Davix::DavixError *err = nullptr;
   if (fFileDes->pos.stat(nullptr, fUrl, &buf, &err) == -1) {
      throw std::runtime_error("Cannot determine size of '" + fUrl + "', error: " + err->getErrMsg());
   }
   return buf.st_size;
}

void ROOT::Detail::RRawFileDavix::Open()
{
   Davix::DavixError *err = nullptr;
   fFileDes->fd = fFileDes->pos.open(nullptr, fUrl, O_RDONLY, &err);
   if (fFileDes->fd == nullptr) {
      throw std::runtime_error("Cannot open '" + fUrl + "', error: " + err->getErrMsg());
   }
}


ROOT::Detail::RRawFileDavix::RRawFileDavix(std::string_view url, RRawFile::ROptions options)
  : RRawFile(url, options), fFileDes(new Internal::RDavixFileDes())
{
}


ROOT::Detail::RRawFileDavix::~RRawFileDavix()
{
   if (fFileDes->fd != nullptr)
      fFileDes->pos.close(fFileDes->fd, nullptr);
}
