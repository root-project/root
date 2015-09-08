/* @(#)root/multiproc:$Id$ */
// Author: Enrico Guiraud July 2015

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_EMPCode
#define ROOT_EMPCode

/////////////////////////////////////////////////////////////////////////
///
/// An enumeration of the message codes handled by the base client and
/// server classes (TMPClient and TMPServer).
///
/////////////////////////////////////////////////////////////////////////

//namespace prevents conflicts with the global variable kError
namespace EMPCode {
   enum EMPCode : unsigned {
      kMessage = 1000,  ///< Generic message
      kError,           ///< Error message
      kFatalError,      ///< Fatal error: whoever sends this message is terminating execution
      kShutdownOrder,   ///< Used by the client to tell servers to shutdown
      kShutdownNotice,  ///< Used by the servers to notify client of shutdown
      kRecvError        ///< MPRecv returns this code when it couldn't read from the socket
   };
}

#endif
