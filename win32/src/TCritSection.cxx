// @(#)root/win32:$Name$:$Id$
// Author: Valery Fine   06/05/96

#include "RConfig.h"
#include "TCritSection.h"

// ClassImp(TCritSection)

//______________________________________________________________________________
TCritSection::TCritSection(){
//*-*
//*-*   Create a critical section object to synchronize threads
//*-*

  flpCriticalSection = (LPCRITICAL_SECTION) malloc(sizeof(CRITICAL_SECTION));
  fSectionCount = 0;
  InitializeCriticalSection(flpCriticalSection);
  fWriteLock = CreateEvent(NULL,TRUE,FALSE,NULL);
}

//______________________________________________________________________________
TCritSection::~TCritSection(){

//*-*  Delete the critial section object

  DeleteCriticalSection(flpCriticalSection);
  free(flpCriticalSection);
  flpCriticalSection = 0;
  CloseHandle(fWriteLock);
}

//______________________________________________________________________________
void TCritSection::WriteLock (){
 top:
    EnterCriticalSection(flpCriticalSection);
    if (fSectionCount) {
        LeaveCriticalSection(flpCriticalSection);
        WaitForSingleObject(fWriteLock,INFINITE);
        goto top;
    }
}

//______________________________________________________________________________
void TCritSection::ReleaseWriteLock(){
    LeaveCriticalSection(flpCriticalSection);
}

//______________________________________________________________________________
void TCritSection::ReadLock(){
    EnterCriticalSection(flpCriticalSection);
    fSectionCount++;
    ResetEvent(fWriteLock);
    LeaveCriticalSection(flpCriticalSection);
}

//______________________________________________________________________________
void TCritSection::ReleaseReadLock(){
    EnterCriticalSection(flpCriticalSection);
    if (--fSectionCount == 0)
           SetEvent(fWriteLock);
    LeaveCriticalSection(flpCriticalSection);
}


