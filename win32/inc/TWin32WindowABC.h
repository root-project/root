/* @(#)root/win32:$Name$:$Id$ */

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/
#ifndef ROOT_TWin32WindowABC
#define ROOT_TWin32WindowABC

#ifndef ROOT_Windows4Root
#include "Windows4Root.h"
#endif

//////////////////////////////////////////////////////////////////////////////
//                                                                          //
//    TWin32WindowABC  is an abstract class to derive those need the Win32  //
//                     Windows objects (ControlBar, Canvas, Pad etc )       //
//                                                                          //
//////////////////////////////////////////////////////////////////////////////


class TWin32WindowABC {

private:

  HWND fHwndWindow;   // Win32 Window HANDLE

public:

  TWin32WindowABC(): fHwndWindow(0){ ; }
  virtual ~TWin32WindowABC();
  virtual void CreateWin32Window() = 0;

  HWND GetWindowHandle(){ return fHwndWindow;}

};

#endif
