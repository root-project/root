// @(#)root/base:$Name:  $:$Id: TCanvasImp.h,v 1.1.1.1 2000/05/16 17:00:39 rdm Exp $
// Author: Fons Rademakers   16/11/95

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/


#ifndef ROOT_TCanvasImp
#define ROOT_TCanvasImp

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TCanvasImp                                                           //
//                                                                      //
// ABC describing GUI independent main window (with menubar, scrollbars //
// and a drawing area).                                                 //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_Rtypes
#include "Rtypes.h"
#endif

class TCanvas;


class TCanvasImp {

protected:
   TCanvas  *fCanvas;   //TCanvas associated with this implementation

public:
   TCanvasImp(TCanvas *c=0) : fCanvas(c) { }
   TCanvasImp(TCanvas *c, const char *name, UInt_t width, UInt_t height);
   TCanvasImp(TCanvas *c, const char *name, Int_t x, Int_t y, UInt_t width, UInt_t height);
   virtual ~TCanvasImp() { }

   TCanvas       *Canvas() const { return fCanvas; }
   virtual void   Close() { }
   virtual void   ForceUpdate() { }
   virtual void   GetWindowGeometry(Int_t &x, Int_t &y, UInt_t &w, UInt_t &h);
   virtual void   Iconify() { }
   virtual Int_t  InitWindow() { return 0; }
   virtual void   SetStatusText(const char *text = 0, Int_t partidx = 0);
   virtual void   SetWindowPosition(Int_t x, Int_t y);
   virtual void   SetWindowSize(UInt_t w, UInt_t h);
   virtual void   SetWindowTitle(const char *newTitle);
   virtual void   SetCanvasSize(UInt_t w, UInt_t h);
   virtual void   ShowMenuBar(Bool_t show = kTRUE);
   virtual void   ShowStatusBar(Bool_t show = kTRUE);
   virtual void   Show() { }

   ClassDef(TCanvasImp,0)  //ABC describing main window protocol
};

inline TCanvasImp::TCanvasImp(TCanvas *, const char *, UInt_t, UInt_t) { }
inline TCanvasImp::TCanvasImp(TCanvas *, const char *, Int_t, Int_t, UInt_t, UInt_t) { }
inline void TCanvasImp::GetWindowGeometry(Int_t &x, Int_t &y, UInt_t &w, UInt_t &h)
               { x = y = 0; w = h = 0; }
inline void TCanvasImp::SetStatusText(const char *, Int_t) { }
inline void TCanvasImp::SetWindowPosition(Int_t, Int_t) { }
inline void TCanvasImp::SetWindowSize(UInt_t, UInt_t) { }
inline void TCanvasImp::SetWindowTitle(const char *) { }
inline void TCanvasImp::SetCanvasSize(UInt_t, UInt_t) { }
inline void TCanvasImp::ShowMenuBar(Bool_t) { }
inline void TCanvasImp::ShowStatusBar(Bool_t) { }

#endif
