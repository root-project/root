// @(#)root/win32:$Name$:$Id$
// Author: Valery Fine   04/03/96

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/


// The class to control WM_COMMAND Win32 messages

#ifndef ROOT_TWin32Command
#define ROOT_TWin32Command

#include "TObjArray.h"

#include "TVirtualMenuItem.h"

class TWin32CommCtrl;

class TWin32Command : public TObjArray {


public:
  TWin32Command(Int_t s=64, Int_t lowerBound=1);
  void ExecuteEvent(Int_t Id,TWin32Canvas *c);
  void JoinMenuItem(TVirtualMenuItem *item);
  void JoinControlItem(TWin32CommCtrl *item);
};

#endif
