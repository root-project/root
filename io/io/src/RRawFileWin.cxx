// @(#)root/io:$Id$
// Author: Jakob Blomer

/*************************************************************************
 * Copyright (C) 1995-2018, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "ROOT/RRawFileWin.hxx"

#include "TError.h"

#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>


ROOT::Detail::RRawFileWin::RRawFileWin(const std::string &url, ROOT::Detail::RRawFile::ROptions options)
  : ROOT::Detail::RRawFile(url, options)
  , fFilePtr(nullptr)
{
}

ROOT::Detail::RRawFileWin::~RRawFileWin()
{
   if (fFilePtr != nullptr)
      fclose(fFilePtr);
}

void ROOT::Detail::RRawFileWin::Seek(long offset, int whence)
{
   int res = fseek(fFilePtr, offset, whence);
   if (res != 0)
      throw std::runtime_error("Cannot seek in '" + fUrl + "', error: " + std::string(strerror(errno)));
}

size_t ROOT::Detail::RRawFileWin::DoReadAt(void *buffer, size_t nbytes, std::uint64_t offset)
{
   if (!IsOpen()) Open();

   Seek(offset, SEEK_SET);
   size_t res = fread(buffer, 1, nbytes, fFilePtr);
   if ((res < nbytes) && (ferror(fFilePtr) != 0)) {
      clearerr(fFilePtr);
      throw std::runtime_error("Cannot read from '" + fUrl + "', error: " + std::string(strerror(errno)));
   }
   return res;
}


std::uint64_t ROOT::Detail::RRawFileWin::DoGetSize()
{
   if (!IsOpen()) Open();

   Seek(0L, SEEK_END);
   long size = ftell(fFilePtr);
   if (size < 0)
     throw std::runtime_error("Cannot tell position counter in '" + fUrl + "', error: " + std::string(strerror(errno)));
   Seek(fFilePos, SEEK_SET);
   return size;
}


void ROOT::Detail::RRawFileWin::Open()
{
   fFilePtr = fopen(GetLocation(fUrl).c_str(), "r");
   if (fFilePtr == nullptr)
      throw std::runtime_error("Cannot open '" + fUrl + "', error: " + std::string(strerror(errno)));
   // Prevent double buffering
   int res = setvbuf(fFilePtr, nullptr, _IONBF, 0);
   R__ASSERT(res == 0);
}
