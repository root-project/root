/* @(#)root/multiproc:$Id$ */
// Author: Enrico Guiraud July 2015

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_MPCode
#define ROOT_MPCode

/////////////////////////////////////////////////////////////////////////
/// This namespace prevents conflicts between MPCode::kError and
/// ELogLevel::kError
namespace MPCode {

   /////////////////////////////////////////////////////////////////////////
   ///
   /// An enumeration of the message codes handled by TMPClient and
   /// TMPWorker.
   ///
   /////////////////////////////////////////////////////////////////////////

   enum EMPCode : unsigned {
      kMessage = 1000,  ///< Generic message
      kError,           ///< Error message
      kFatalError,      ///< Fatal error: whoever sends this message is terminating execution
      kShutdownOrder,   ///< Used by the client to tell servers to shutdown
      kShutdownNotice,  ///< Used by the workers to notify client of shutdown
      kRecvError        ///< Error while reading from the socket
   };
}

#endif
