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
   static constexpr std::uint64_t kUnknownFileSize = std::uint64_t(-1);
   struct ROptions {
      ROptions()  { }
   };
   enum class ELineBreaks { kSystem, kUnix, kWindows };

private:
   static const char* kTransportSeparator;

protected:
   std::string fUrl;
   ROptions fOptions;
   std::uint64_t fFilePos;
   std::uint64_t fFileSize;

   static std::string GetLocation(std::string_view url);
   static std::string GetTransport(std::string_view url);

   virtual size_t DoPread(void *buffer, size_t nbytes, std::uint64_t offset) = 0;
   virtual std::uint64_t DoGetSize() = 0;

public:
   // TODO: Move constructor, copy/assignment delete
   RRawFile(const std::string &url, ROptions options);
   virtual ~RRawFile();
   static RRawFile* Create(std::string_view url, ROptions options = ROptions());

   size_t Pread(void *buffer, size_t nbytes, std::uint64_t offset);
   size_t Read(void *buffer, size_t nbytes);
   void Seek(std::uint64_t offset);
   std::uint64_t GetSize();

   std::string Readln(ELineBreaks lineBreak = ELineBreaks::kSystem);
};


class RRawFilePosix : public RRawFile {
private:
   int filedes;

   void EnsureOpen();

protected:
   size_t DoPread(void *buffer, size_t nbytes, std::uint64_t offset) final;
   std::uint64_t DoGetSize() final;

public:
   RRawFilePosix(const std::string &url, RRawFile::ROptions options);
   ~RRawFilePosix();
};

} // namespace Detail
} // namespace ROOT

#endif
