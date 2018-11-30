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

#include "TError.h"

#include <algorithm>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>

namespace {
const char* kTransportSeparator = "://";
// Corresponds to ELineBreaks
#ifdef _WIN32
const char* kLineBreakTokens[] = {"", "\r\n", "\n", "\r\n"};
constexpr unsigned int kLineBreakTokenSizes[] = {0, 2, 1, 2};
#else
const char* kLineBreakTokens[] = {"", "\n", "\n", "\r\n"};
constexpr unsigned int kLineBreakTokenSizes[] = {0, 1, 1, 2};
#endif
constexpr unsigned int kLineBuffer = 128; // On Readln, look for line-breaks in chunks of 128 bytes
constexpr int kDefaultBlockSizeLocal = 4096; // Local files are by default read in pages
} // anonymous namespace


size_t ROOT::Detail::RRawFile::RBlockBuffer::Map(void *buffer, size_t nbytes, std::uint64_t offset)
{
   if (offset < fBufferOffset)
      return 0;

   size_t mappedBytes = 0;
   std::uint64_t offsetInBuffer = offset - fBufferOffset;
   if (offsetInBuffer < static_cast<std::uint64_t>(fBufferSize)) {
      size_t bytesInBuffer = std::min(nbytes, static_cast<size_t>(fBufferSize - offsetInBuffer));
      memcpy(buffer, fBuffer + offsetInBuffer, bytesInBuffer);
      mappedBytes = bytesInBuffer;
   }
   return mappedBytes;
}


ROOT::Detail::RRawFile::RRawFile(
   const std::string &url,
   ROOT::Detail::RRawFile::ROptions options)
   : fBlockBufferIdx(0)
   , fBufferSpace(nullptr)
   , fFileSize(kUnknownFileSize)
   , fUrl(url)
   , fOptions(options)
   , fFilePos(0)
{
   if (fOptions.fBlockSize > 0) {
      fBufferSpace = new unsigned char[kNumBlockBuffers * fOptions.fBlockSize];
      for (unsigned int i = 0; i < kNumBlockBuffers; ++i)
         fBlockBuffers[i].fBuffer = fBufferSpace + i * fOptions.fBlockSize;
   }
}


ROOT::Detail::RRawFile::~RRawFile() {
   delete[] fBufferSpace;
}


bool ROOT::Detail::RRawFile::Readln(std::string& line)
{
   if (fOptions.fLineBreak == ELineBreaks::kAuto) {
      // Auto-detect line breaks according to the break discovered in the first line
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
         // Line break found, return the string and skip the linebreak itself
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
      if (options.fBlockSize < 0)
        options.fBlockSize = kDefaultBlockSizeLocal;
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
   R__ASSERT(fOptions.fBlockSize >= 0);

   // "Large" reads are served directly, bypassing the cache
   if (nbytes > static_cast<unsigned int>(fOptions.fBlockSize))
      return DoPread(buffer, nbytes, offset);

   size_t totalBytes = 0;
   size_t mappedBytes = 0;
   /// Try to serve as many bytes as possible from the block buffers
   for (unsigned int idx = fBlockBufferIdx; idx < fBlockBufferIdx + kNumBlockBuffers; ++idx) {
      mappedBytes = fBlockBuffers[idx % kNumBlockBuffers].Map(buffer, nbytes, offset);
      buffer = reinterpret_cast<unsigned char *>(buffer) + mappedBytes;
      nbytes -= mappedBytes;
      offset += mappedBytes;
      totalBytes += mappedBytes;
      if (mappedBytes > 0)
         fBlockBufferIdx = idx;
      if (nbytes == 0)
         return totalBytes;
   }
   fBlockBufferIdx++;

   /// The request was not fully satisfied and fBlockBufferIdx now points to the previous shadow buffer

   /// The remaining bytes populate the newly promoted main buffer
   RBlockBuffer* thisBuffer = &fBlockBuffers[fBlockBufferIdx % kNumBlockBuffers];
   size_t res = DoPread(thisBuffer->fBuffer, fOptions.fBlockSize, offset);
   thisBuffer->fBufferOffset = offset;
   thisBuffer->fBufferSize = res;
   size_t remainingBytes = std::min(res, nbytes);
   memcpy(buffer, thisBuffer->fBuffer, remainingBytes);
   totalBytes += remainingBytes;
   return totalBytes;
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
