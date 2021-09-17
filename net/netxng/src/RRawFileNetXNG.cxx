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

#include <memory>
#include <stdexcept>
#include <vector>
#include <XrdCl/XrdClFile.hh>
#include <XrdCl/XrdClFileSystem.hh>


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
   if( fOptions.fBlockSize < 0 ) fOptions.fBlockSize = kDefaultBlockSize;
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

