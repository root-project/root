// @(#)root/thread:$Name:  $:$Id: TProof.h,v 1.113 2007/05/25 13:53:59 ganis Exp $
// Author: Jan Fiete Grosse-Oetringhaus, 04.06.07

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TLockFile
#define ROOT_TLockFile

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TLockFile                                                            //
//                                                                      //
// Lock an object using a file.                                         //
// Constructor blocks until lock is obtained. Lock is released in the   //
// destructor.                                                          //
//                                                                      //
// Use it in scope-blocks like:                                         //
// {                                                                    //
//    TLockFile lock("path.to.lock.file");                              //
//    // do something you need the lock for                             //
// } // lock is automatically released                                  //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TObject
#include "TObject.h"
#endif
#ifndef ROOT_TString
#include "TString.h"
#endif


class TLockFile : public TObject
{
private:
   TLockFile(const TLockFile&);             // not implemented
   TLockFile& operator=(const TLockFile&);  // not implemented

protected:
   TString fPath;         // path to file holding the lock

   Bool_t Lock(const char* path, Int_t timeLimit);

public:
    TLockFile(const char* path, Int_t timeLimit = 0);
    virtual ~TLockFile();

    ClassDef(TLockFile, 0)
};

#endif

