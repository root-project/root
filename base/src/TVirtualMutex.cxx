// @(#)root/base:$Name:  $:$Id: TVirtualMutex.cxx,v 1.1 2002/02/14 16:12:52 rdm Exp $
// Author: Fons Rademakers   14/02/2002

/*************************************************************************
 * Copyright (C) 1995-2002, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TVirtualMutex                                                        //
//                                                                      //
// This class implements a mutex interface. The actual work is done via //
// TMutex which is available as soon as the thread library is loaded.   //
//                                                                      //
// and                                                                  //
//                                                                      //
// TLockGuard                                                           //
//                                                                      //
// This class provides mutex resource management in a guaranteed and    //
// exception safe way. Use like this:                                   //
// {                                                                    //
//    TLockGuard guard(mutex);                                          //
//    ... // do something                                               //
// }                                                                    //
// when guard goes out of scope the mutex is unlocked in the TLockGuard //
// destructor. The exception mechanism takes care of calling the dtors  //
// of local objects so it is exception safe.                            //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TVirtualMutex.h"

ClassImp(TVirtualMutex)
ClassImp(TLockGuard)


TVirtualMutex *gContainerMutex = 0;
TVirtualMutex *gCINTMutex = 0;
