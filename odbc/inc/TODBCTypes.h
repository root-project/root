// @(#)root/odbc:$Name:  $:$Id: TODBCRow.h,v 1.1 2006/04/17 14:12:52 rdm Exp $
// Author: Fons Rademakers   18/04/2006

/*************************************************************************
 * Copyright (C) 1995-2006, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TODBCTypes
#define ROOT_TODBCTypes

#ifndef ROOT_Rtypes
#include "Rtypes.h"
#endif

// Microsoft is confused about sizes on different platforms.
#ifdef R__B64
typedef UInt_t  ODBCUInt_t;
typedef Int_t   ODBCInt_t;
#else
typedef ULong_t ODBCUInt_t;
typedef Long_t  ODBCInt_t;
#endif

#endif

