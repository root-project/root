/*************************************************************************
 * Copyright (C) 1995-2020, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_RIoUring
#define ROOT_RIoUring

#include <cstring>
#include <stdexcept>
#include <vector>

#include <liburing.h>
#include <liburing/io_uring.h>

#include <ROOT/RRawFile.hxx>

#include "TError.h"

namespace ROOT {
namespace Internal {

class RIoUring {
private:
   struct io_uring fRing;
   size_t fSize;

   static bool CheckIsAvailable() {
      try {
         RIoUring(1);
         return true;
      }
      catch (const std::runtime_error& err) {
         Warning("RIoUring", "io_uring is not available\n%s", err.what());
      }
      return false;
   }

public:
   explicit RIoUring(size_t size) : fSize(size) {
      int ret = io_uring_queue_init(fSize, &fRing, 0 /* no flags */);
      if (ret) {
         throw std::runtime_error("Error initializing io_uring: " + std::string(std::strerror(-ret)));
      }
   }

   RIoUring(const RIoUring&) = delete;
   RIoUring& operator=(const RIoUring&) = delete;


   ~RIoUring() {
      // todo(max) try submitting any pending events before exiting
      io_uring_queue_exit(&fRing);
   }

   /// Check if io_uring is available on this system.
   static bool IsAvailable() {
      static const bool available = RIoUring::CheckIsAvailable();
      return available;
   }

   /// Access the raw io_uring instance.
   struct io_uring *GetRawRing() {
      return &fRing;
   }

   /// Basic read event composed of RIOVec IO data and a target file descriptor.
   struct RReadEvent {
      RRawFile::RIOVec* fIoVec = nullptr;
      int fFileDes = -1;
   };

   /// Submit a number of read events and wait for completion.
   void SubmitReadsAndWait(std::vector<RReadEvent>& readEvents) {
      auto numEvents = readEvents.size();
      if (numEvents > fSize) {
         throw std::runtime_error("too many read events (" + std::to_string(numEvents) + ") for "
            + "ring with size (" + std::to_string(fSize) + "). event batching is not implemented yet");
      }

      // todo(max) think about registering fFileDes to avoid repeated kernel fd mappings
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
      for (std::size_t i = 0; i < numEvents; ++i) {
         sqe = io_uring_get_sqe(&fRing);
         if (!sqe) {
            throw std::runtime_error("get SQE failed for read request '" +
               std::to_string(i) + "', error: " + std::string(strerror(errno)));
         }
         if (readEvents[i].fFileDes == -1) {
            throw std::runtime_error("bad fd (-1) for read request '" + std::to_string(i) + "'");
         }
         if (readEvents[i].fIoVec == nullptr) {
            throw std::runtime_error("null RIOVec* for read request '" + std::to_string(i) + "'");
         }
         io_uring_prep_read(sqe,
            readEvents[i].fFileDes,
            readEvents[i].fIoVec->fBuffer,
            readEvents[i].fIoVec->fSize,
            readEvents[i].fIoVec->fOffset
         );
         sqe->user_data = i;
      }

      // todo(max) fix for batched sqe submissions where ret may not equal nReq
      // todo(max) check for any difference between submit vs. submit and wait for large nReq
      int submitted = io_uring_submit_and_wait(&fRing, numEvents);
      if (submitted <= 0) {
         throw std::runtime_error("ring submit failed, error: " + std::string(strerror(errno)));
      }
      if (submitted != static_cast<int>(numEvents)) {
         throw std::runtime_error("ring submitted " + std::to_string(submitted) +
            " events but requested " + std::to_string(numEvents));
      }
      // reap reads
      struct io_uring_cqe *cqe;
      int ret;
      for (int i = 0; i < submitted; ++i) {
         ret = io_uring_wait_cqe(&fRing, &cqe);
         if (ret < 0) {
            throw std::runtime_error("wait cqe failed, error: " + std::string(std::strerror(-ret)));
         }
         auto index = reinterpret_cast<std::size_t>(io_uring_cqe_get_data(cqe));
         if (index >= numEvents) {
            throw std::runtime_error("bad cqe user data: " + std::to_string(index));
         }
         if (cqe->res < 0) {
            throw std::runtime_error("read failed for ReadEvent[" + std::to_string(index) + "], "
               "error: " + std::string(std::strerror(-cqe->res)));
         }
         readEvents[index].fIoVec->fOutBytes = static_cast<std::size_t>(cqe->res);
         io_uring_cqe_seen(&fRing, cqe);
      }
      return;
   }
};

} // namespace Internal
} // namespace ROOT

#endif
