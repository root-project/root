// @(#)root/gui:$Id$
// Author: Fons Rademakers   22/02/98

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TRootControlBar
#define ROOT_TRootControlBar


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TRootControlBar                                                      //
//                                                                      //
// This class provides an interface to the GUI dependent functions of   //
// the TControlBar class. A control bar is a horizontal or vertical bar //
// with a number of buttons (text or picture buttons).                  //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TControlBarImp
#include "TControlBarImp.h"
#endif
#ifndef ROOT_TGFrame
#include "TGFrame.h"
#endif

class TControlBar;
class TList;


class TRootControlBar : public TGMainFrame, public TControlBarImp {

private:
   TList          *fWidgets; // list of TGTextButton or TGPictureButtons
   TGLayoutHints  *fL1;      // button layout hints
   UInt_t          fBwidth;  // button width in pixels

public:
   TRootControlBar(TControlBar *c = 0, const char *title = "ROOT Control Bar",
                   Int_t x = -999, Int_t y = -999);
   virtual ~TRootControlBar();

   void Create();
   void Hide();
   void Show();

   TList *GetWidgets() const { return fWidgets; }

   // overridden from TGMainFrame
   void   CloseWindow();
   Bool_t ProcessMessage(Long_t msg, Long_t parm1, Long_t parm2);
   void   ReallyDelete();
   void   SetButtonState(const char *label, Int_t state = 0);
   void   SetButtonWidth(UInt_t width);
   void   SetFont(const char *fontName);
   void   SetTextColor(const char *colorName);

   ClassDef(TRootControlBar,0)  //ROOT native GUI implementation of TControlBar
};

#endif
