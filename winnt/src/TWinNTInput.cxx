// @(#)root/winnt:$Name$:$Id$
// Author: Fons Rademakers   02/04/95

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TWinNTInput                                                          //
//                                                                      //
// This class encapsulates input callback from file descriptors.        //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TWinNTInput.h"
#include "TApplication.h"


ClassImp(TWinNTInput)


TWinNTInput::TWinNTInput()
{
   fFd = -1;
}

TWinNTInput::~TWinNTInput()
{
}

 void TWinNTInput::Attach(int fd, int mask)
 {
   if (fFd != -1)
      Remove();

   fFd = fd;

   AllocConsole();
}

void TWinNTInput::Remove()
{
   if (fFd != -1)
          FreeConsole();
   fFd = -1;
}

int TWinNTInput::Fd()
{
   return fFd;
}

int TWinNTInput::Id()
{
     return 0;
}

void TWinNTInput::InputHandler(void * clientData, int *, void *)
{
   TWinNTInput *obj = (TWinNTInput*)clientData;

}

