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

#include <liburing.h>
#include <liburing/io_uring.h>

#include "TError.h"

namespace ROOT {
namespace Internal {

class RIoUring {
private:
   struct io_uring fRing;

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
   explicit RIoUring(size_t size) {
      int ret = io_uring_queue_init(size, &fRing, 0 /* no flags */);
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
};

} // namespace Internal
} // namespace ROOT

#endif
