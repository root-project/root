// @(#)root/win32:$Name$:$Id$
// Author: Valery Fine   19/01/96

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/
#ifndef ROOT_TGWin32Pen
#define ROOT_TGWin32Pen

#ifndef ROOT_TObject
#include "TObject.h"
#endif

#ifndef ROOT_Windows4Root
#include "Windows4Root.h"
#endif

class TGWin32Pen : public TObject {

private:

   HPEN       fhdPen;
   LOGPEN     fPen;
   LOGBRUSH   fBrush;
   int        flUserDash;
   int       *fUserDash;

public:

   TGWin32Pen();
   ~TGWin32Pen();
   void Delete();
   HPEN CreatePen();
   COLORREF GetColor(){return fBrush.lbColor;}
   void SetWidth(Width_t width=1);
   void SetType(int n=0, int *dash = 0);
   void SetColor(COLORREF cindex=1);
   HGDIOBJ GetWin32Pen();

   // ClassDef(TGWin32Pen,0)  // Pen class for Win32 interface
};

#endif
