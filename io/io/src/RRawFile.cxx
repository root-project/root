// @(#)root/io:$Id$
// Author: Jakob Blomer

/*************************************************************************
 * Copyright (C) 1995-2018, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "ROOT/RRawFile.hxx"

#include "TError.h"

#include <cerrno>
#include <cstring>
#include <stdexcept>

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

namespace ROOT {
namespace Detail {

const char* ROOT::Detail::RRawFile::kTransportSeparator = "://";

ROOT::Detail::RRawFile::RRawFile(
   const std::string &url,
   ROOT::Detail::RRawFile::ROptions options)
   : fUrl(url)
   , fOptions(options)
   , fFilePos(0)
   , fFileSize(kUnknownFileSize)
{
}


ROOT::Detail::RRawFile::~RRawFile()
{
}

std::string ROOT::Detail::RRawFile::Readln(ROOT::Detail::RRawFile::ELineBreaks /*lineBreak*/)
{
   return "";
}

ROOT::Detail::RRawFile* ROOT::Detail::RRawFile::Create(std::string_view url, ROptions options)
{
   std::string transport = GetTransport(url);
	if (transport == "file") {
		return new RRawFilePosix(std::string(url), options);
	}
	return nullptr;
}

std::string ROOT::Detail::RRawFile::GetLocation(std::string_view url)
{
	auto idx = url.find(kTransportSeparator);
   if (idx == std::string_view::npos)
	  return std::string(url);
   return std::string(url.substr(idx + strlen(kTransportSeparator)));
}

std::string ROOT::Detail::RRawFile::GetTransport(std::string_view url)
{
	auto idx = url.find(kTransportSeparator);
	if (idx == std::string_view::npos)
	   return "file";
	return std::string(url.substr(0, idx));
}


size_t ROOT::Detail::RRawFile::Pread(void *buffer, size_t nbytes, std::uint64_t offset)
{
   return DoPread(buffer, nbytes, offset);
}

size_t ROOT::Detail::RRawFile::Read(void *buffer, size_t nbytes)
{
   return Pread(buffer, nbytes, fFilePos);
}

void ROOT::Detail::RRawFile::Seek(std::uint64_t offset)
{
  fFilePos = offset;
}

std::uint64_t ROOT::Detail::RRawFile::GetSize()
{
   if (fFileSize == kUnknownFileSize)
      fFileSize = DoGetSize();
   return fFileSize;
}


/******************************************************************************/

ROOT::Detail::RRawFilePosix::RRawFilePosix(const std::string &url, ROOT::Detail::RRawFile::ROptions options)
  : ROOT::Detail::RRawFile(url, options)
  , filedes(-1)
{
}

ROOT::Detail::RRawFilePosix::~RRawFilePosix()
{
   if (filedes >= 0)
	   close(filedes);
}

size_t ROOT::Detail::RRawFilePosix::DoPread(void *buffer, size_t nbytes, std::uint64_t offset)
{
   EnsureOpen();

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


std::uint64_t ROOT::Detail::RRawFilePosix::DoGetSize()
{
	EnsureOpen();
   struct stat info;
	int res = fstat(filedes, &info);
	if (res == 0)
	  return info.st_size;
	throw std::runtime_error("Cannot call fstat on '" + fUrl + "', error: " + std::string(strerror(errno)));
}


void ROOT::Detail::RRawFilePosix::EnsureOpen()
{
   if (filedes >= 0)
      return;

   filedes = open(GetLocation(fUrl).c_str(), O_RDONLY);
   if (filedes < 0) {
      throw std::runtime_error("Cannot open '" + fUrl + "', error: " + std::string(strerror(errno)));
   }
}


} // namespace Detail
} // namespace ROOT
