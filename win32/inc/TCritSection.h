// @(#)root/win32:$Name$:$Id$
// Author: Valery Fine   06/05/96

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/
#ifndef ROOT_TCritSection
#define ROOT_TCritSection

#ifndef ROOT_Windows4Root
#include "Windows4Root.h"
#endif

#ifndef ROOT_TObject
#include "TObject.h"
#endif

class TCritSection : public TObject {

private:
   LPCRITICAL_SECTION  flpCriticalSection; // pointer to critical section object
   int          fSectionCount;             // flag to mark whether we are witin the section
   HANDLE       fhEvent;                   // The event object to synch threads
   HANDLE       fWriteLock;                // Event object to synch thread


public:

    TCritSection();
    ~TCritSection();

    void      WriteLock ();
    void      ReleaseWriteLock();
    void      ReadLock();
    void      ReleaseReadLock();


    void      XW_OpenSemaphore(){;}
    void      XW_CloseSemaphore(){;}
    void      XW_WaitSemaphore(){;}
    void      XW_CreateSemaphore(){;}

    // ClassDef(TCritSection,0)
};
#endif
