// @(#)root/net:$Id$
// Author: Jakob Blomer

/*************************************************************************
 * Copyright (C) 1995-2025, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_RRawFileCurl
#define ROOT_RRawFileCurl

#include <ROOT/RRawFile.hxx>

#include <memory>

namespace ROOT {
namespace Internal {

class RCurlConnection;

/// Provides read-only access to files on HTTP(S) resources. Uses libcurl for the transport layer.
class RRawFileCurl : public RRawFile {
private:
   std::unique_ptr<RCurlConnection> fCurlConnection;

protected:
   void OpenImpl() final;
   size_t ReadAtImpl(void *buffer, size_t nbytes, std::uint64_t offset) final;
   std::uint64_t GetSizeImpl() final;

public:
   RRawFileCurl(std::string_view url, RRawFile::ROptions options);
   ~RRawFileCurl();
   std::unique_ptr<RRawFile> Clone() const final;
};

} // namespace Internal
} // namespace ROOT

#endif
