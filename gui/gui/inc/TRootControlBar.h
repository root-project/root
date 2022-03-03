// @(#)root/gui:$Id$
// Author: Fons Rademakers   22/02/98

/*************************************************************************
 * Copyright (C) 1995-2021, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TRootControlBar
#define ROOT_TRootControlBar


#include "TControlBarImp.h"
#include "TGFrame.h"

class TControlBar;
class TList;


class TRootControlBar : public TGMainFrame, public TControlBarImp {

private:
   TList          *fWidgets; ///< list of TGTextButton or TGPictureButtons
   TGLayoutHints  *fL1;      ///< button layout hints
   UInt_t          fBwidth;  ///< button width in pixels

public:
   TRootControlBar(TControlBar *c = nullptr, const char *title = "ROOT Control Bar",
                   Int_t x = -999, Int_t y = -999);
   virtual ~TRootControlBar();

   void Create();
   void Hide();
   void Show();

   TList *GetWidgets() const { return fWidgets; }

   // overridden from TGMainFrame
   void   CloseWindow();
   Bool_t ProcessMessage(Longptr_t msg, Longptr_t parm1, Longptr_t parm2);
   void   ReallyDelete();
   void   SetButtonState(const char *label, Int_t state = 0);
   void   SetButtonWidth(UInt_t width);
   void   SetFont(const char *fontName);
   void   SetTextColor(const char *colorName);

   ClassDef(TRootControlBar,0)  //ROOT native GUI implementation of TControlBar
};

#endif
