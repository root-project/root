// @(#)root/io:$Id$
// Author: Jakob Blomer

/*************************************************************************
 * Copyright (C) 1995-2018, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "ROOT/RConfig.hxx"
#include "ROOT/RRawFileUnix.hxx"
#include "ROOT/RMakeUnique.hxx"

#ifdef R__HAS_URING
  #include "ROOT/RIoUring.hxx"
#endif

#include "TError.h"

#include <cerrno>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace {
constexpr int kDefaultBlockSize = 4096; // If fstat() does not provide a block size hint, use this value instead
} // anonymous namespace

ROOT::Internal::RRawFileUnix::RRawFileUnix(std::string_view url, ROptions options)
   : RRawFile(url, options), fFileDes(-1)
{
}

ROOT::Internal::RRawFileUnix::~RRawFileUnix()
{
   if (fFileDes >= 0)
      close(fFileDes);
}

std::unique_ptr<ROOT::Internal::RRawFile> ROOT::Internal::RRawFileUnix::Clone() const
{
   return std::make_unique<RRawFileUnix>(fUrl, fOptions);
}

int ROOT::Internal::RRawFileUnix::GetFeatures() const {
   return kFeatureHasSize | kFeatureHasMmap;
}

std::uint64_t ROOT::Internal::RRawFileUnix::GetSizeImpl()
{
#ifdef R__SEEK64
   struct stat64 info;
   int res = fstat64(fFileDes, &info);
#else
   struct stat info;
   int res = fstat(fFileDes, &info);
#endif
   if (res != 0)
      throw std::runtime_error("Cannot call fstat on '" + fUrl + "', error: " + std::string(strerror(errno)));
   return info.st_size;
}

void *ROOT::Internal::RRawFileUnix::MapImpl(size_t nbytes, std::uint64_t offset, std::uint64_t &mapdOffset)
{
   static std::uint64_t szPageBitmap = sysconf(_SC_PAGESIZE) - 1;
   mapdOffset = offset & ~szPageBitmap;
   nbytes += offset & szPageBitmap;

   void *result = mmap(nullptr, nbytes, PROT_READ, MAP_PRIVATE, fFileDes, mapdOffset);
   if (result == MAP_FAILED)
      throw std::runtime_error(std::string("Cannot perform memory mapping: ") + strerror(errno));
   return result;
}

void ROOT::Internal::RRawFileUnix::OpenImpl()
{
#ifdef R__SEEK64
   fFileDes = open64(GetLocation(fUrl).c_str(), O_RDONLY);
#else
   fFileDes = open(GetLocation(fUrl).c_str(), O_RDONLY);
#endif
   if (fFileDes < 0) {
      throw std::runtime_error("Cannot open '" + fUrl + "', error: " + std::string(strerror(errno)));
   }

   if (fOptions.fBlockSize >= 0)
      return;

#ifdef R__SEEK64
   struct stat64 info;
   int res = fstat64(fFileDes, &info);
#else
   struct stat info;
   int res = fstat(fFileDes, &info);
#endif
   if (res != 0) {
      throw std::runtime_error("Cannot call fstat on '" + fUrl + "', error: " + std::string(strerror(errno)));
   }
   if (info.st_blksize > 0) {
      fOptions.fBlockSize = info.st_blksize;
   } else {
      fOptions.fBlockSize = kDefaultBlockSize;
   }
}

void ROOT::Internal::RRawFileUnix::ReadVImpl(RIOVec *ioVec, unsigned int nReq)
{
#ifdef R__HAS_URING
   if (RIoUring::IsAvailable()) {
      RIoUring ring(nReq);
      struct io_uring *p_ring = ring.raw();

      // todo(max) try registering fFileDes to avoid repeated kernel fd mappings
      // ```
      // io_uring_register_files(p_ring, &fFileDes, 1);
      // -- then fd parameter in prep_read is the offset into the array of fixed files
      // -- (i.e. 0, because we only have one file)
      // io_uring_prep_read(sqe, 0, ioVec[i].fBuffer, ioVec[i].fSize, ioVec[i].fOffset);
      // -- and the sqe flags have to be adjusted
      // sqe->flags |= IOSQE_FIXED_FILE;
      // ```
      // files are unregistered when the queue is destroyed

      // prep reads
      struct io_uring_sqe *sqe;
      for (std::size_t i = 0; i < nReq; ++i) {
         sqe = io_uring_get_sqe(p_ring);
         if (!sqe) {
            throw std::runtime_error("get SQE failed for read request '" +
               std::to_string(i) + "', error: " + std::string(strerror(errno)));
         }
         io_uring_prep_read(sqe, fFileDes, ioVec[i].fBuffer, ioVec[i].fSize, ioVec[i].fOffset);
         sqe->user_data = i;
      }

      // todo(max) fix for batched sqe submissions where ret may not equal nReq
      int submitted = io_uring_submit_and_wait(p_ring, nReq);
      if (submitted <= 0) {
         throw std::runtime_error("ring submit failed, error: " + std::string(strerror(errno)));
      }
      if (submitted != static_cast<int>(nReq)) {
         throw std::runtime_error("ring submitted " + std::to_string(submitted) +
            " events but requested " + std::to_string(nReq));
      }
      // reap reads
      struct io_uring_cqe *cqe;
      int ret;
      for (int i = 0; i < submitted; ++i) {
         ret = io_uring_wait_cqe(p_ring, &cqe);
         if (ret < 0) {
            throw std::runtime_error("wait cqe failed, error: " + std::string(std::strerror(-ret)));
         }
         auto index = reinterpret_cast<std::size_t>(io_uring_cqe_get_data(cqe));
         if (index >= nReq) {
            throw std::runtime_error("bad cqe user data: " + std::to_string(index));
         }
         if (cqe->res < 0) {
            throw std::runtime_error("read failed for RIOVec[" + std::to_string(index) + "], "
               "error: " + std::string(std::strerror(-cqe->res)));
         }
         ioVec[index].fOutBytes = static_cast<std::size_t>(cqe->res);
         io_uring_cqe_seen(p_ring, cqe);
      }
      return;
   }
   Warning("RRawFileUnix",
           "io_uring setup failed, falling back to default ReadV implementation");
#endif
   RRawFile::ReadVImpl(ioVec, nReq);
}

size_t ROOT::Internal::RRawFileUnix::ReadAtImpl(void *buffer, size_t nbytes, std::uint64_t offset)
{
   size_t total_bytes = 0;
   while (nbytes) {
#ifdef R__SEEK64
      ssize_t res = pread64(fFileDes, buffer, nbytes, offset);
#else
      ssize_t res = pread(fFileDes, buffer, nbytes, offset);
#endif
      if (res < 0) {
         if (errno == EINTR)
            continue;
         throw std::runtime_error("Cannot read from '" + fUrl + "', error: " + std::string(strerror(errno)));
      } else if (res == 0) {
         return total_bytes;
      }
      R__ASSERT(static_cast<size_t>(res) <= nbytes);
      buffer = reinterpret_cast<unsigned char *>(buffer) + res;
      nbytes -= res;
      total_bytes += res;
      offset += res;
   }
   return total_bytes;
}

void ROOT::Internal::RRawFileUnix::UnmapImpl(void *region, size_t nbytes)
{
   int rv = munmap(region, nbytes);
   if (rv != 0)
      throw std::runtime_error(std::string("Cannot remove memory mapping: ") + strerror(errno));
}
