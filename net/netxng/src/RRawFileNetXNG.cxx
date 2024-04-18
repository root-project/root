// @(#)root/io:$Id$
// Author: Michal Simon

/*************************************************************************
 * Copyright (C) 1995-2018, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "ROOT/RRawFileNetXNG.hxx"

#include <TError.h>

#include <cctype>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <utility>
#include <XrdCl/XrdClFile.hh>
#include <XrdCl/XrdClFileSystem.hh>
#include <XrdVersion.hh>

namespace {
constexpr int kDefaultBlockSize = 128 * 1024; // Read in relatively large 128k blocks for better network utilization
} // anonymous namespace

namespace ROOT {
namespace Internal {

struct RRawFileNetXNGImpl {
   RRawFileNetXNGImpl() = default;
   RRawFileNetXNGImpl(const RRawFileNetXNGImpl &) = delete;
   RRawFileNetXNGImpl &operator=(const RRawFileNetXNGImpl &) = delete;
   ~RRawFileNetXNGImpl() = default;

   XrdCl::File file;
};

} // namespace Internal
} // namespace ROOT


ROOT::Internal::RRawFileNetXNG::RRawFileNetXNG( std::string_view   url,
                                                RRawFile::ROptions options )
   : RRawFile( url, options ), pImpl( new RRawFileNetXNGImpl() )
{
}

ROOT::Internal::RRawFileNetXNG::~RRawFileNetXNG()
{
}

std::unique_ptr<ROOT::Internal::RRawFile> ROOT::Internal::RRawFileNetXNG::Clone() const
{
   return std::make_unique<RRawFileNetXNG>( fUrl, fOptions );
}

std::uint64_t ROOT::Internal::RRawFileNetXNG::GetSizeImpl()
{
   XrdCl::StatInfo *info = nullptr;
   auto st = pImpl->file.Stat( true, info );
   if( !st.IsOK() )
     throw std::runtime_error( "Cannot determine size of '" + fUrl + "', " +
                               st.ToString() + "; " + st.GetErrorMessage() );
   std::uint64_t ret = info->GetSize();
   delete info; // XrdCl only allocates the object is the status was OK
   return ret;
}

void ROOT::Internal::RRawFileNetXNG::OpenImpl()
{
   auto st = pImpl->file.Open( fUrl, XrdCl::OpenFlags::Read );
   if( !st.IsOK() )
     throw std::runtime_error( "Cannot open '" + fUrl + "', " +
                               st.ToString() + "; " + st.GetErrorMessage() );
   if (fOptions.fBlockSize == ROptions::kUseDefaultBlockSize)
      fOptions.fBlockSize = kDefaultBlockSize;
}

size_t ROOT::Internal::RRawFileNetXNG::ReadAtImpl(void *buffer, size_t nbytes, std::uint64_t offset)
{
   std::uint32_t btsread = 0;
   auto st = pImpl->file.Read( offset, nbytes, buffer, btsread );
   if( !st.IsOK() )
     throw std::runtime_error( "Cannot read from '" + fUrl + "', " +
                               st.ToString() + "; " + st.GetErrorMessage() );
   return btsread;
}

void ROOT::Internal::RRawFileNetXNG::ReadVImpl(RIOVec *ioVec, unsigned int nReq)
{
   XrdCl::ChunkList chunks;
   chunks.reserve( nReq );
   for( std::size_t i = 0; i < nReq; ++i )
     chunks.emplace_back( ioVec[i].fOffset, ioVec[i].fSize, ioVec[i].fBuffer );

   XrdCl::VectorReadInfo *info = nullptr;
   auto st = pImpl->file.VectorRead( chunks, nullptr, info );
   if( !st.IsOK() )
     throw std::runtime_error( "Cannot do vector read from '" + fUrl + "', " +
                               st.ToString() + "; " + st.GetErrorMessage() );

   XrdCl::ChunkList &rsp = info->GetChunks();
   for( std::size_t i = 0; i < nReq; ++i )
     ioVec[i].fOutBytes = rsp[i].length;
   delete info;
}

ROOT::Internal::RRawFile::RIOVecLimits ROOT::Internal::RRawFileNetXNG::GetReadVLimits()
{
   if (fIOVecLimits)
      return *fIOVecLimits;

   EnsureOpen();
   // Start with xrootd default values
   fIOVecLimits = RIOVecLimits{1024, 2097136, static_cast<std::uint64_t>(-1)};

#if XrdVNUMBER >= 40000
   std::string strLastURL;
   pImpl->file.GetProperty("LastURL", strLastURL);
   XrdCl::URL lastURL(strLastURL);
   // local redirect will split vector reads into multiple local reads anyway,
   // so we are fine with the default values
   if (lastURL.GetProtocol().compare("file") == 0 && lastURL.GetHostId().compare("localhost") == 0) {
      if (gDebug >= 1)
         Info("GetReadVLimits", "Local redirect, using default values");
      return *fIOVecLimits;
   }

   std::string strDataServer;
   if (!pImpl->file.GetProperty("DataServer", strDataServer)) {
      if (gDebug >= 1)
         Info("GetReadVLimits", "Cannot get DataServer property, using default values");
      return *fIOVecLimits;
   }
   XrdCl::URL dataServer(strDataServer);
#else
   XrdCl::URL dataServer(pImpl->file.GetDataServer());
#endif

   XrdCl::FileSystem fs(dataServer);
   XrdCl::Buffer arg;
   XrdCl::Buffer *response = nullptr;
   arg.FromString("readv_ior_max readv_iov_max");

   XrdCl::XRootDStatus status = fs.Query(XrdCl::QueryCode::Config, arg, response);
   if (!status.IsOK()) {
      delete response;
      if (gDebug >= 1)
         Info("GetReadVLimits", "Cannot query readv limits, using default values");
      return *fIOVecLimits;
   }
   std::istringstream strmResponse;
   strmResponse.str(response->ToString());
   delete response;

   std::string readvMaxSingleSize;
   std::string readvMaxReqs;
   if (!std::getline(strmResponse, readvMaxSingleSize) || !std::getline(strmResponse, readvMaxReqs)) {
      if (gDebug >= 1)
         Info("GetReadVLimits", "unexpected response from querying readv limits, using default values");
      return *fIOVecLimits;
   }

   if (!readvMaxReqs.empty() && std::isdigit(readvMaxReqs[0])) {
      std::size_t val = std::stoi(readvMaxReqs);
      // Workaround a dCache bug reported here: https://sft.its.cern.ch/jira/browse/ROOT-6639
      if (val == 0x7FFFFFFF)
         return *fIOVecLimits;

      fIOVecLimits->fMaxReqs = val;
   }

   if (!readvMaxSingleSize.empty() && std::isdigit(readvMaxSingleSize[0])) {
      fIOVecLimits->fMaxSingleSize = std::stoi(readvMaxSingleSize);
   }

   return *fIOVecLimits;
}
