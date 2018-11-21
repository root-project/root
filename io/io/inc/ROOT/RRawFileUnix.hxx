// @(#)root/io:$Id$
// Author: Jakob Blomer

/*************************************************************************
 * Copyright (C) 1995-2018, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_RRawFileUnix
#define ROOT_RRawFileUnix

#include <ROOT/RRawFile.hxx>

#include <cstddef>
#include <cstdint>
#include <string>

namespace ROOT {
namespace Detail {

class RRawFileUnix : public RRawFile {
private:
   int filedes;
   void EnsureOpen();

protected:
   size_t DoPread(void *buffer, size_t nbytes, std::uint64_t offset) final;
   std::uint64_t DoGetSize() final;

public:
   RRawFileUnix(const std::string &url, RRawFile::ROptions options);
   ~RRawFileUnix();
};

} // namespace Detail
} // namespace ROOT

#endif
