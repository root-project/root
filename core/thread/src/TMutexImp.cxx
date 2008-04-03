// @(#)root/thread:$Id$
// Author: Fons Rademakers   01/07/97

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TMutexImp                                                            //
//                                                                      //
// This class provides an abstract interface to the OS dependent mutex  //
// classes (TPosixMutex and TWin32Mutex).                               //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TMutexImp.h"

ClassImp(TMutexImp)
