// @(#)root/gl:$Id$
// Author:  Matevz Tadel, Feb 2007

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGLLockable_H
#define ROOT_TGLLockable_H

#include <Rtypes.h>

class TGLLockable
{
public:
   enum ELock { kUnlocked,    // Unlocked
                kDrawLock,    // Locked for draw, cannot select or modify
                kSelectLock,  // Locked for select, cannot modify (draw part of select)
                kModifyLock   // Locked for modify, cannot draw or select
              };

private:
   TGLLockable(const TGLLockable&);            // Not implemented
   TGLLockable& operator=(const TGLLockable&); // Not implemented

protected:
   // Locking - can take/release via const handle
   mutable ELock     fLock;        // Lock state.

   // Ensures unlocking in view of exceptions.
   class TUnlocker
   {
   private:
      TUnlocker(const TUnlocker&);            // Not implemented
      TUnlocker& operator=(const TUnlocker&); // Not implemented

      const TGLLockable *fLockable;

   public:
      TUnlocker(const TGLLockable* l) : fLockable(l) {}
      ~TUnlocker()
      {
         if (fLockable->IsLocked())
            fLockable->ReleaseLock(fLockable->CurrentLock());
      }
   };

public:
   TGLLockable();
   virtual ~TGLLockable() {}

   virtual const char* LockIdStr() const { return "<unknown>"; }

   Bool_t TakeLock(ELock lock)    const;
   Bool_t ReleaseLock(ELock lock) const;
   Bool_t IsLocked()    const { return (fLock != kUnlocked); }
   ELock  CurrentLock() const { return fLock; }

   Bool_t IsDrawOrSelectLock() const { return fLock == kDrawLock || fLock == kSelectLock; }

   static const char * LockName(ELock lock);
   static Bool_t       LockValid(ELock lock);

   ClassDef(TGLLockable, 0); // Lock for viewers and scenes.
}; // endclass TGLLockable

#endif
