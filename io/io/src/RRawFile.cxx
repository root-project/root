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

namespace {
const char* kTransportSeparator = "://";
#ifdef _WIN32
const char* kLineBreakTokens[] = {"", "\r\n", "\n", "\r\n"};
constexpr unsigned kLineBreakTokenSizes[] = {0, 2, 1, 2};
#else
const char* kLineBreakTokens[] = {"", "\n", "\n", "\r\n"};
constexpr unsigned kLineBreakTokenSizes[] = {0, 1, 1, 2};
#endif
constexpr unsigned kLineBuffer = 128;
} // anonymous namespace


ROOT::Detail::RRawFile::RRawFile(
   const std::string &url,
   ROOT::Detail::RRawFile::ROptions options)
   : fUrl(url)
   , fOptions(options)
   , fFilePos(0)
   , fFileSize(kUnknownFileSize)
{
}


bool ROOT::Detail::RRawFile::Readln(std::string& line)
{
   line.clear();
   if (fOptions.fLineBreak == ELineBreaks::kAuto) {
      fOptions.fLineBreak = ELineBreaks::kUnix;
      bool res = Readln(line);
      if ((line.length() > 0) && (*line.rbegin() == '\r')) {
         fOptions.fLineBreak = ELineBreaks::kWindows;
         line.resize(line.length() - 1);
      }
      return res;
   }

   char buffer[kLineBuffer];
   size_t nbytes;
   do {
      nbytes = Read(buffer, sizeof(buffer));
      std::string_view bufferView(buffer, nbytes);
      auto idx = bufferView.find(kLineBreakTokens[static_cast<int>(fOptions.fLineBreak)]);
      if (idx != std::string_view::npos) {
         // Linebreak found, return the string and skip the linebreak itself
         line.append(buffer, idx);
         fFilePos += kLineBreakTokenSizes[static_cast<int>(fOptions.fLineBreak)];
         break;
      }
      line.append(buffer, nbytes);
   } while (nbytes > 0);

   return nbytes > 0;
}


ROOT::Detail::RRawFile* ROOT::Detail::RRawFile::Create(std::string_view url, ROptions options)
{
   std::string transport = GetTransport(url);
   if (transport == "file") {
      return new RRawFilePosix(std::string(url), options);
   }
   throw std::runtime_error("Unsupported transport protocol: " + transport);
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

/******************************************************************************/

ROOT::Detail::RRawFileCio::RRawFileCio(const std::string &url, ROOT::Detail::RRawFile::ROptions options)
  : ROOT::Detail::RRawFile(url, options)
  , fileptr(nullptr)
{
}

ROOT::Detail::RRawFileCio::~RRawFileCio()
{
   if (fileptr != nullptr)
      fclose(fileptr);
}

void ROOT::Detail::RRawFileCio::Seek(long offset, int whence)
{
   int res = fseek(fileptr, offset, whence);
   if (res != 0)
      throw std::runtime_error("Cannot seek in '" + fUrl + "', error: " + std::string(strerror(errno)));
}

size_t ROOT::Detail::RRawFileCio::DoPread(void *buffer, size_t nbytes, std::uint64_t offset)
{
   EnsureOpen();
   if (offset != fFilePos)
      Seek(offset, SEEK_SET);
   size_t res = fread(buffer, 1, nbytes, fileptr);
   if ((res < nbytes) && (ferror(fileptr) != 0)) {
      clearerr(fileptr);
      throw std::runtime_error("Cannot read from '" + fUrl + "', error: " + std::string(strerror(errno)));
   }
   return res;
}


std::uint64_t ROOT::Detail::RRawFileCio::DoGetSize()
{
   EnsureOpen();
   Seek(0L, SEEK_END);
   long size = ftell(fileptr);
   if (size < 0)
     throw std::runtime_error("Cannot tell position counter in '" + fUrl + "', error: " + std::string(strerror(errno)));
   Seek(fFilePos, SEEK_SET);
   return size;
}


void ROOT::Detail::RRawFileCio::EnsureOpen()
{
   if (fileptr != nullptr)
      return;

   fileptr = fopen(GetLocation(fUrl).c_str(), "r");
   if (fileptr == nullptr)
      throw std::runtime_error("Cannot open '" + fUrl + "', error: " + std::string(strerror(errno)));
}


} // namespace Detail
} // namespace ROOT
