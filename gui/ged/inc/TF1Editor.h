// @(#)root/ged:$Id$
// Author: Ilka Antcheva 21/03/06

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TF1Editor
#define ROOT_TF1Editor


#include "TGedFrame.h"

class TGNumberEntry;
class TGTextEntry;
class TGTextButton;
class TGDoubleHSlider;
class TGNumberEntryField;
class TGLabel;
class TGCheckButton;
class TF1;

class TF1Editor : public TGedFrame {

protected:
   TF1                 *fF1;              ///< selected TF1 object
   TGTextEntry         *fTitle;           ///< function title
   Int_t                fNP;              ///< number of function parameters
   TGLabel             *fParLabel;        ///< label for number of parameters
   TGCheckButton       *fDrawMode;        ///< immediate function redraw (if selected)
   TGTextButton        *fSetPars;         ///< open 'Set Parameters' dialog
   TGNumberEntry       *fNXpoints;        ///< number of points along x-axis
   TGDoubleHSlider     *fSliderX;         ///< slider to set x-axis range
   TGNumberEntryField  *fSldMinX;         ///< contains minimum value of x-axis
   TGNumberEntryField  *fSldMaxX;         ///< contains maximum value of x-axis

   virtual void ConnectSignals2Slots();   //connect signals to slots

public:
   TF1Editor(const TGWindow *p = 0,  Int_t width = 140, Int_t height = 30,
             UInt_t options = kChildFrame, Pixel_t back = GetDefaultFrameBackground());
   virtual ~TF1Editor();

   virtual void   SetModel(TObject* obj);
   virtual void   ActivateBaseClassEditors(TClass* cl);

   virtual void   DoParameterSettings();
   virtual void   DoXPoints();
   virtual void   DoSliderXMoved();
   virtual void   DoSliderXPressed();
   virtual void   DoSliderXReleased();
   virtual void   DoXRange();

   ClassDef(TF1Editor,0)  // user interface for TF1 objects
};

#endif
