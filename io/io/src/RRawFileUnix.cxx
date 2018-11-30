// @(#)root/io:$Id$
// Author: Jakob Blomer

/*************************************************************************
 * Copyright (C) 1995-2018, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "ROOT/RRawFileUnix.hxx"

#include "TError.h"

#include <cerrno>
#include <cstring>
#include <stdexcept>

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>


ROOT::Detail::RRawFileUnix::RRawFileUnix(const std::string &url, ROOT::Detail::RRawFile::ROptions options)
  : ROOT::Detail::RRawFile(url, options)
  , filedes(-1)
{
}


ROOT::Detail::RRawFileUnix::~RRawFileUnix()
{
   if (filedes >= 0)
      close(filedes);
}


size_t ROOT::Detail::RRawFileUnix::DoReadAt(void *buffer, size_t nbytes, std::uint64_t offset)
{
   if (!IsOpen()) Open();

   size_t total_bytes = 0;
   while (nbytes) {
      ssize_t res = pread(filedes, buffer, nbytes, offset);
      if (res < 0) {
         if (errno == EINTR)
            continue;
         throw std::runtime_error("Cannot read from '" + fUrl + "', error: " + std::string(strerror(errno)));
      } else if (res == 0) {
         return total_bytes;
      }
      R__ASSERT(static_cast<size_t>(res) <= nbytes);
      buffer = reinterpret_cast<unsigned char *>(buffer) + res;
      nbytes -= res;
      total_bytes += res;
      offset += res;
   }
   return total_bytes;
}


std::uint64_t ROOT::Detail::RRawFileUnix::DoGetSize()
{
   if (!IsOpen()) Open();
   struct stat info;
   int res = fstat(filedes, &info);
   if (res != 0)
     throw std::runtime_error("Cannot call fstat on '" + fUrl + "', error: " + std::string(strerror(errno)));
   return info.st_size;
}


void ROOT::Detail::RRawFileUnix::Open()
{
   filedes = open(GetLocation(fUrl).c_str(), O_RDONLY);
   if (filedes < 0) {
      throw std::runtime_error("Cannot open '" + fUrl + "', error: " + std::string(strerror(errno)));
   }
}
