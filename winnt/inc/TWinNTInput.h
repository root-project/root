// @(#)root/winnt:$Name$:$Id$
// Author: Fons Rademakers   02/04/95

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef CodeROOT_TWinNTInput
#define CodeROOT_TWinNTInput


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TWinNTInput                                                          //
//                                                                      //
// This class encapsulates input callback from file descriptors.        //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#if !defined(__CINT__)
#include <windows.h>
#endif

#ifndef ROOT_Rtypes
#include "Rtypes.h"
#endif


class TWinNTInput {

private:

  static void InputHandler(void *clientData, int *source, void *id);

protected:

   int       fFd;

 public:

   TWinNTInput();
  ~TWinNTInput();
   void Attach(int fd, int mask);
   void Remove();
   int Fd();
   int Id();

   ClassDef(TWinNTInput,0)  //Handle callbacks from file descriptors
};

#endif
