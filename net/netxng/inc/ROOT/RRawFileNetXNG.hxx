// @(#)root/io:$Id$
// Author: Michal Simon

/*************************************************************************
 * Copyright (C) 1995-2018, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef NET_NETXNG_INC_ROOT_RRAWFILENETXNG_HXX_
#define NET_NETXNG_INC_ROOT_RRAWFILENETXNG_HXX_

#include <ROOT/RRawFile.hxx>
#include <memory>

namespace ROOT {
namespace Internal {

struct RRawFileNetXNGImpl;

/** \class RRawFileNetXNG RRawFileNetXNG.hxx

The RRawFileNetXNG class provides read-only access to remote files using root/roots protocol. It uses the
XrdCl (XRootD client) library for the transport layer.  It instructs the RRawFile base class to buffer in
larger chunks than the default for local files, assuming that remote file access has high(er) latency.

*/

class RRawFileNetXNG : public RRawFile
{
private:
   std::unique_ptr<RRawFileNetXNGImpl> pImpl; //< pointer to implementation

protected:
   void OpenImpl() final;
   size_t ReadAtImpl(void *buffer, size_t nbytes, std::uint64_t offset) final;
   void ReadVImpl(RIOVec *ioVec, unsigned int nReq) final;
   std::uint64_t GetSizeImpl() final;

public:
   RRawFileNetXNG(std::string_view url, RRawFile::ROptions options);
   ~RRawFileNetXNG();
   std::unique_ptr<RRawFile> Clone() const final;
   int GetFeatures() const final { return kFeatureHasSize | kFeatureHasAsyncIo; }
};

} // namespace Internal
} // namespace ROOT

#endif /* NET_NETXNG_INC_ROOT_RRAWFILENETXNG_HXX_ */
