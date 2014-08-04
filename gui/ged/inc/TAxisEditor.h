// @(#)root/ged:$Id$
// Author: Ilka  Antcheva 11/05/04

/*************************************************************************
 * Copyright (C) 1995-2002, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TAxisEditor
#define ROOT_TAxisEditor

//////////////////////////////////////////////////////////////////////////
//                                                                      //
//  TAxisEditor                                                         //
//                                                                      //
//  Implements GUI for axis attributes.                                 //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TGedFrame
#include "TGedFrame.h"
#endif

class TAxis;
class TGLabel;
class TGComboBox;
class TGNumberEntry;
class TGTextEntry;
class TGCheckButton;
class TGColorSelect;
class TGFontTypeComboBox;


class TAxisEditor : public TGedFrame {

protected:
   TAxis               *fAxis;         // axis object
   TGColorSelect       *fAxisColor;    // color selection widget
   TGCheckButton       *fLogAxis;      // logarithmic check box
   TGNumberEntry       *fTickLength;   // tick length number entry
   TGNumberEntry       *fDiv1;         // primary axis division number entry
   TGNumberEntry       *fDiv2;         // secondary axis division number entry
   TGNumberEntry       *fDiv3;         // tertiary axis division number entry
   TGCheckButton       *fOptimize;     // tick optimization check box
   TGCheckButton       *fTicksBoth;    // check box setting ticks on both axis sides
   TGCheckButton       *fMoreLog;      // more logarithmic labels check box
   Int_t                fTicksFlag;    // positive/negative ticks' flag
   TGTextEntry         *fTitle;        // axis title input field
   TGColorSelect       *fTitleColor;   // color selection widget
   TGFontTypeComboBox  *fTitleFont;    // title font combo box
   Int_t                fTitlePrec;    // font precision level
   TGNumberEntry       *fTitleSize;    // title size number entry
   TGNumberEntry       *fTitleOffset;  // title offset number entry
   TGCheckButton       *fCentered;     // check button for centered title
   TGCheckButton       *fRotated;      // check button for rotated title
   TGColorSelect       *fLabelColor;   // color selection widget
   TGFontTypeComboBox  *fLabelFont;    // label font combo box
   Int_t                fLabelPrec;    // font precision level
   TGNumberEntry       *fLabelSize;    // label size number entry
   TGNumberEntry       *fLabelOffset;  // label offset number entry
   TGCheckButton       *fNoExponent;   // check box for No exponent choice
   TGCheckButton       *fDecimal;      // decimal part check box

   virtual void ConnectSignals2Slots();

public:
   TAxisEditor(const TGWindow *p = 0,
               Int_t width = 140, Int_t height = 30,
               UInt_t options = kChildFrame,
               Pixel_t back = GetDefaultFrameBackground());
   virtual ~TAxisEditor();
   virtual void   SetModel(TObject* obj);
   // slots related to axis attributes
   virtual void   DoTickLength();
   virtual void   DoAxisColor(Pixel_t color);
   virtual void   DoTicks();
   virtual void   DoDivisions();
   virtual void   DoLogAxis();
   virtual void   DoMoreLog();
   // slots related to axis title attributes
   virtual void   DoTitleColor(Pixel_t color);
   virtual void   DoTitle(const char *text);
   virtual void   DoTitleSize();
   virtual void   DoTitleFont(Int_t font);
   virtual void   DoTitleOffset();
   virtual void   DoTitleCentered();
   virtual void   DoTitleRotated();
   // slots related to axis labels attributes
   virtual void   DoLabelColor(Pixel_t color);
   virtual void   DoLabelSize();
   virtual void   DoLabelFont(Int_t font);
   virtual void   DoLabelOffset();
   virtual void   DoNoExponent();
   virtual void   DoDecimal(Bool_t on);

   ClassDef(TAxisEditor,0)  // axis editor
};

#endif
