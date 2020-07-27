/*************************************************************************
 * Copyright (C) 1995-2020, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_RIoUring
#define ROOT_RIoUring

#include <liburing.h>
#include <liburing/io_uring.h>

#include <ROOT/RError.hxx>
using ROOT::Experimental::RException;

namespace ROOT {
namespace Internal {

class RIoUring {
private:
   struct io_uring fRing;
public:
   explicit RIoUring(size_t size) {
      int ret = io_uring_queue_init(size, &fRing, 0 /* no flags */);
      if (ret) {
         throw RException(R__FAIL("couldn't open ring"));
      }
   }

   RIoUring(const RIoUring&) = delete;
   RIoUring& operator=(const RIoUring&) = delete;

   ~RIoUring() {
      io_uring_queue_exit(&fRing);
   }

   /// Check if io_uring is available on this system.
   static bool IsAvailable() {
      thread_local bool couldOpenRing = false;
      thread_local bool firstPass = true;
      if (firstPass) {
         firstPass = false;
         try {
            RIoUring(1);
            couldOpenRing = true;
         }
         catch (const RException&) {
            // ring setup failed
         }
      }
      return couldOpenRing;
   }
};

} // namespace Internal
} // namespace ROOT

#endif
