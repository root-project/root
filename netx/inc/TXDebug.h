// @(#)root/netx:$Name:  $:$Id: TXDebug.h,v 1.2 2004/08/20 22:16:33 rdm Exp $
// Author: Alvise Dorigo, Fabrizio Furano

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TXDebug
#define ROOT_TXDebug

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TXDebug                                                              //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TEnv
#include "TEnv.h"
#endif

#define DebugLevel() gXDebugLevel

enum EXDebugLevel {
   kNODEBUG   = 0,
   kUSERDEBUG = 1,
   kHIDEBUG   = 2,
   kDUMPDEBUG = 3
};

R__EXTERN Short_t gXDebugLevel;

#endif
