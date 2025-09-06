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

/// \class RRawFileCurl
/// \ingroup net
///
/// The RRawFileCurl class reads HTTP(S) resources using the curl library. The passed URL of the file
/// needs to start with http:// or https://. The URL does not need to be escaped; URL encoding is handled internally.
class RRawFileCurl : public RRawFile {
private:
   std::unique_ptr<RCurlConnection> fConnection;

protected:
   void OpenImpl() final;
   size_t ReadAtImpl(void *buffer, size_t nbytes, std::uint64_t offset) final;
   void ReadVImpl(RIOVec *ioVec, unsigned int nReq) final;
   std::uint64_t GetSizeImpl() final;

public:
   static constexpr int kDefaultBlockSize = 128 * 1024; // relatively large 128k blocks for better network utilization

   RRawFileCurl(std::string_view url, RRawFile::ROptions options);
   ~RRawFileCurl();
   std::unique_ptr<RRawFile> Clone() const final;

   RCurlConnection &GetConnection();
};

} // namespace Internal
} // namespace ROOT

#endif
