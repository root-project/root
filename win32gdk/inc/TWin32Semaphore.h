/* @(#)root/win32gdk:$Name:$:$Id:$ */

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/
#ifndef ROOT_TWin32Semaphore
#define ROOT_TWin32Semaphore

#include <winbase.h>

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
