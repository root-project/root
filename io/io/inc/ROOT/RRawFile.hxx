// @(#)root/io:$Id$
// Author: Jakob Blomer

/*************************************************************************
 * Copyright (C) 1995-2018, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_RRawFile
#define ROOT_RRawFile

#include <ROOT/RStringView.hxx>

#include <cstddef>
#include <cstdint>
#include <string>

namespace ROOT {
namespace Detail {

/**
 * \class RRawFile RRawFile.hxx
 * \ingroup IO
 *
 * Read-only, usually remote file
 */
class RRawFile {
public:
   class ROptions {
   };

   enum class ELineBreaks { kLbSystem, kLbUnix, kLbWindows };

protected:
   std::string fProtocol;
   std::string fLocation;
   ROptions fOptions;
   std::uint64_t fFilePos;

   RRawFile(const std::string &fProtocol, const std::string &fLocation, ROptions options);

public:
   // TODO: Move constructor, copy/assignment delete
   virtual ~RRawFile();
   static RRawFile* Create(std::string_view url, ROptions options = ROptions());

   virtual ssize_t Pread(void *buffer, ssize_t nbytes, std::uint64_t offset) = 0;
   virtual ssize_t Read(void *buffer, ssize_t nbytes) = 0;
   virtual std::string Readln(ELineBreaks lineBreak = ELineBreaks::kLbSystem) = 0;
};



} // namespace Detail
} // namespace ROOT

#endif
