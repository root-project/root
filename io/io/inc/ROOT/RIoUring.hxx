/*************************************************************************
 * Copyright (C) 1995-2020, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_RIoUring
#define ROOT_RIoUring

#define HAS_LIBURING

extern "C" {
  #include <liburing.h>
  #include <liburing/io_uring.h>
}

#include <ROOT/RError.hxx>

namespace ROOT {
namespace Internal {

class IoUring {
private:
   struct io_uring fRing;
public:
   explicit IoUring(size_t size) {
      int ret = io_uring_queue_init(size, &fRing, 0 /* no flags */);
      if (ret) {
         throw R__FAIL("couldn't open ring");
      }
   }
   ~IoUring() {
      io_uring_queue_exit(&fRing);
   }
   // todo try opening a ring and check the error code
   static bool io_uring_available() {
      return false; 
   }
}; 

} // namespace Internal
} // namespace ROOT

#endif
