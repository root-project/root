// @(#)root/ged:$Name:  $:$Id: TAxisEditor.h,v 1.0 2004/05/11 16:28:28 brun Exp $
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
//  TAttAxisEditor                                                      //
//                                                                      //
//  Editor of axis attributes. It contains three frames related         //
//  to the axis (TGedAxisFrame), axis title (TGedAxisTitle) and         //
//  axis label (TGedAxisLabel) attributes.                              //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TGButton
#include "TGWidget.h"
#endif
#ifndef ROOT_TGedFrame
#include "TGedFrame.h"
#endif
#ifndef ROOT_TAxis
#include "TAxis.h"
#endif

class TGLabel;
class TGComboBox;
class TGNumberEntry;
class TGTextEntry;
class TGCheckButton;
class TGColorSelect;
class TGFontTypeComboBox;


//////////////////////////////////////////////////////////////////////////
//                                                                      //
//  TAxisEditor                                                         //
//                                                                      //
//  Editor related to axis, axis title and axis label attributes.       //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

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

public:
   TAxisEditor(const TGWindow *p, Int_t id,
               Int_t width = 140, Int_t height = 30,
               UInt_t options = kChildFrame,
               Pixel_t back = GetDefaultFrameBackground());
   virtual ~TAxisEditor();
   virtual void   SetModel(TVirtualPad *pad, TObject *obj, Int_t event);
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


   ClassDef(TAxisEditor,0)  // axis editor
};

#endif
