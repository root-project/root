// @(#)root/proof:$Id$
// Author: G. Ganis, Oct 2015

/*************************************************************************
 * Copyright (C) 1995-2005, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TLockPath
#define ROOT_TLockPath

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TLockPath                                                            //
//                                                                      //
// Path locking class allowing shared and exclusive locks               //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TObject.h"
#include "TString.h"

class TLockPath : public TObject {
private:
   TString       fName;          // path to lock
   Int_t         fLockId;        // file id of dir lock

public:
   TLockPath(const char *path = "");
   ~TLockPath() { if (IsLocked()) Unlock(); }

   const char   *GetName() const { return fName; }
   void          SetName(const char *path) { fName = path; }

   Int_t         Lock(Bool_t shared = kFALSE);
   Int_t         Unlock();

   Bool_t        IsLocked() const { return (fLockId > -1); }

   ClassDef(TLockPath, 0)  // Path locking class
};

class TLockPathGuard {
private:
   TLockPath  *fLocker; //locker instance

public:
   TLockPathGuard(TLockPath *l, Bool_t shared = kFALSE) {
                                fLocker = l; fLocker->Lock(shared); }
   ~TLockPathGuard() { fLocker->Unlock(); }
};

#endif
