// @(#)root/rpdutils:$Id$
// Author: Gerardo Ganis   7/4/2003

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_rpddefs
#define ROOT_rpddefs


/////////////////////////////////////////////////////////////////////
//                                                                 //
// Definition used by daemons                                      //
//                                                                 //
/////////////////////////////////////////////////////////////////////

#include "AuthConst.h"

//
// Typedefs
typedef void (*SigPipe_t)(int);

//
// Global consts
const int  kMAXRECVBUF       = 1024;
const int  kMAXUSERLEN       = 128;

// Masks for initialization options
const unsigned int kDMN_RQAUTH = 0x1;  // Require authentication
const unsigned int kDMN_HOSTEQ = 0x2;  // Allow host equivalence
const unsigned int kDMN_SYSLOG = 0x4;  // Log messages to syslog i.o. stderr

//
// type of service
enum  EService  { kSOCKD = 0, kROOTD, kPROOFD };

#endif
