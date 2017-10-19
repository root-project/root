// @(#)root/proof:$Id$
// Author: G. Ganis, Oct 2011

/*************************************************************************
 * Copyright (C) 1995-2005, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

/** \class TLockPath
\ingroup proofkernel

Path locking class allowing shared and exclusive locks

*/

#include "TLockPath.h"
#include "TSystem.h"
#if defined(R__WIN32) && !defined(R__WINGCC)
#include <io.h>
#define lseek _lseek
#define close _close
#define open _open
#define O_CREAT _O_CREAT
#define O_RDWR _O_RDWR
#else
#include <sys/file.h>
#endif

////////////////////////////////////////////////////////////////////////////////
/// Locks the directory. Waits if lock is hold by an other process.
/// Returns 0 on success, -1 in case of error.

TLockPath::TLockPath(const char *path) : fName(path), fLockId(-1)
{
   // Work with full names
   if (gSystem->ExpandPathName(fName))
      Warning("TLockPath", "problems expanding path '%s'", fName.Data());
}

Int_t TLockPath::Lock(Bool_t shared)
{
   const char *pname = GetName();

   if (gSystem->AccessPathName(pname))
      fLockId = open(pname, O_CREAT | O_RDWR, 0644);
   else
      fLockId = open(pname, O_RDWR);

   if (fLockId == -1) {
      SysError("Lock", "cannot open lock file %s", pname);
      return -1;
   }

   if (gDebug > 1)
      Info("Lock", "%d: locking file %s ...", gSystem->GetPid(), pname);
   // lock the file
#if !defined(R__WIN32) && !defined(R__WINGCC)
   int op = (shared) ? LOCK_SH : LOCK_EX ;
   if (flock(fLockId, op) == -1) {
      SysError("Lock", "error locking %s", pname);
      close(fLockId);
      fLockId = -1;
      return -1;
   }
#endif

   if (gDebug > 1)
      Info("Lock", "%d: file %s locked", gSystem->GetPid(), pname);

   return 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Unlock the directory. Returns 0 in case of success,
/// -1 in case of error.

Int_t TLockPath::Unlock()
{
   if (!IsLocked())
      return 0;

   if (gDebug > 1)
      Info("Unlock", "%d: unlocking file %s ...", gSystem->GetPid(), GetName());
   // unlock the file
   lseek(fLockId, 0, SEEK_SET);
#if !defined(R__WIN32) && !defined(R__WINGCC)
   if (flock(fLockId, LOCK_UN) == -1) {
      SysError("Unlock", "error unlocking %s", GetName());
      close(fLockId);
      fLockId = -1;
      return -1;
   }
#endif

   if (gDebug > 1)
      Info("Unlock", "%d: file %s unlocked", gSystem->GetPid(), GetName());

   close(fLockId);
   fLockId = -1;

   return 0;
}
