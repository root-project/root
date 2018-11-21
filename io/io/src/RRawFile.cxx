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
#include "ROOT/RRawFileDavix.hxx"
#ifdef _WIN32
#include "ROOT/RRawFileWin.hxx"
#else
#include "ROOT/RRawFileUnix.hxx"
#endif

#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>

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
   if (fOptions.fLineBreak == ELineBreaks::kAuto) {
      fOptions.fLineBreak = ELineBreaks::kUnix;
      bool res = Readln(line);
      if ((line.length() > 0) && (*line.rbegin() == '\r')) {
         fOptions.fLineBreak = ELineBreaks::kWindows;
         line.resize(line.length() - 1);
      }
      return res;
   }

   line.clear();
   char buffer[kLineBuffer];
   size_t nbytes;
   do {
      nbytes = Read(buffer, sizeof(buffer));
      std::string_view bufferView(buffer, nbytes);
      auto idx = bufferView.find(kLineBreakTokens[static_cast<int>(fOptions.fLineBreak)]);
      if (idx != std::string_view::npos) {
         // Linebreak found, return the string and skip the linebreak itself
         line.append(buffer, idx);
         fFilePos -= nbytes - idx;
         fFilePos += kLineBreakTokenSizes[static_cast<int>(fOptions.fLineBreak)];
         return true;
      }
      line.append(buffer, nbytes);
   } while (nbytes > 0);

   return !line.empty();
}


ROOT::Detail::RRawFile* ROOT::Detail::RRawFile::Create(std::string_view url, ROptions options)
{
   std::string transport = GetTransport(url);
   if (transport == "file") {
#ifdef _WIN32
      return new RRawFileWin(std::string(url), options);
#else
      return new RRawFileUnix(std::string(url), options);
#endif
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
   size_t res = Pread(buffer, nbytes, fFilePos);
   fFilePos += res;
   return res;
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
