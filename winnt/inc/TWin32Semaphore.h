/* @(#)root/winnt:$Name:  $:$Id: TWin32Semaphore.h,v 1.1.1.1 2000/05/16 17:00:47 rdm Exp $ */

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TWin32Semaphore
#define ROOT_TWin32Semaphore

#include <Windows4Root.h>

class TWin32Semaphore {

private:
  HANDLE  fSemaphore;  // Handle of the semaphore to synch actitvities

public:
  TWin32Semaphore(){fSemaphore = CreateSemaphore(NULL, 0, 1, NULL);}
  virtual ~TWin32Semaphore(){ CloseHandle(fSemaphore); }
  virtual void Wait() {WaitForSingleObject(fSemaphore, INFINITE);}
  virtual void Release(){ReleaseSemaphore(fSemaphore,1L,NULL); }
};


#endif
