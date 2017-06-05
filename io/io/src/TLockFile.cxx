// @(#)root/io:$Id$
// Author: Jan Fiete Grosse-Oetringhaus, 04.06.07

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

/**
\class TLockFile
\ingroup IO

A scoped lock based on files.

The RAAI idiom is used: the constructor blocks until lock is obtained.
Lock is released in the destructor.
Use it in scope-blocks like:
~~~{.cpp}
{
   TLockFile lock("path.to.lock.file");
   // do something you need the lock for
} // lock is automatically released
~~~
*/

#include "TLockFile.h"
#include "TSystem.h"
#include "TFile.h"
#include <time.h>

ClassImp(TLockFile);

////////////////////////////////////////////////////////////////////////////////
/// Default constructor.
///
/// Blocks until lock is obtained.
/// If a lock exists that is older than the given time limit,
/// the file is removed. If timeLimit <= 0, wait for ever.

TLockFile::TLockFile(const char *path, Int_t timeLimit) : fPath(path)
{
   while (1) {
      if (Lock(fPath, timeLimit))
         break;

      if (gDebug > 0)
         Info("TLockFile", "did not aquire lock %s, sleeping...", fPath.Data());
      gSystem->Sleep(1000);
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Destructor. Releases the lock.

TLockFile::~TLockFile()
{
   if (gDebug > 0)
      Info("~TLockFile", "releasing lock %s", fPath.Data());

   gSystem->Unlink(fPath);
}

////////////////////////////////////////////////////////////////////////////////
/// Internal function that locks with the given path.

Bool_t TLockFile::Lock(const char *path, Int_t timeLimit)
{
   Long_t modTime = 0;
   if (gSystem->GetPathInfo(path, 0, (Long_t*) 0, 0, &modTime) == 0) {
      if (timeLimit > 0) {
         if (gDebug > 0)
            Info("Lock", "%s modification time %ld, %ld seconds ago", path, modTime, time(0) - modTime);
         if (time(0) - modTime > timeLimit){
            gSystem->Unlink(path);
            if (gDebug > 0)
               Info("Lock", "time expired, removed %s", path);
         } else
            return kFALSE;
      } else
         return kFALSE;
   }

   TString spath = path;
   spath += "?filetype=raw";
   TFile *file = TFile::Open(spath, "CREATE");
   if (!file)
      return kFALSE;

   file->Close();
   delete file;

   // chance access to 666, so if the lock is expired, other users can remove it
   // (attention, currently only supported for local files systems)
   gSystem->Chmod(path, 0666);

   if (gDebug > 0)
      Info("Lock", "obtained lock %s", path);

   return kTRUE;
}
