// @(#)root/netx:$Name:  $:$Id: TNetFile.h,v 1.16 2004/08/09 17:43:07 rdm Exp $
// Author: Alvise Dorigo, Fabrizio Furano

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TXMutexLocker
#define ROOT_TXMutexLocker

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TXMutexLocker                                                        //
//                                                                      //
// Authors: Alvise Dorigo, Fabrizio Furano                              //
//          INFN Padova, 2003                                           //
//                                                                      //
//////////////////////////////////////////////////////////////////////////



class TXMutexLocker {

private:
   TMutex *fMtx;

public:

   inline TXMutexLocker(TMutex *mutex) {
      fMtx = mutex;
      fMtx->Lock();
   }
   inline ~TXMutexLocker() { fMtx->UnLock(); }
};




#endif
