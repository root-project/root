// @(#)root/net:$Name:$:$Id:$
// Author: Fons Rademakers   28/08/2003

/*************************************************************************
 * Copyright (C) 1995-2003, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// NetErrors                                                            //
//                                                                      //
// This file defines error strings mapped to the error codes generated  //
// by rootd/proofd.                                                     //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "NetErrors.h"

// Must match order of ERootdErrors enum in NetErrors.h
const char *gRootdErrStr[] = {
   "undefined error",
   "file not found",
   "error in file name",
   "file already exists",
   "no access to file",
   "error opening file",
   "file already opened in read or write mode",
   "file already opened in write mode",
   "no more space on device",
   "bad op code",
   "bad message",
   "error writing to file",
   "error reading from file",
   "no such user",
   "remote not setup for anonymous access",
   "illegal user name",
   "can't cd to home directory",
   "can't get passwd info",
   "wrong passwd",
   "no SRP support in remote daemon",
   "fatal error",
   "cannot seek to restart position",
   "server does not accept the requested authentication method from this host or from user@host",
   "server does not accept connection from this host: contact server administrator",
   "authentication attempt unsuccessful"
};
