// @(#)root/ged:$Name:  $:$Id: TAttMarkerEditor.h,v 1.1 2004/06/18 15:55:00 brun Exp $
// Author: Ilka  Antcheva 11/05/04

/*************************************************************************
 * Copyright (C) 1995-2002, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TAttMarkerEditor
#define ROOT_TAttMarkerEditor

//////////////////////////////////////////////////////////////////////////
//                                                                      //
//  TAttMarkerEditor                                                    //
//                                                                      //
//  Implements GUI for editing marker attributes.                       //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TGButton
#include "TGWidget.h"
#endif
#ifndef ROOT_TGedFrame
#include "TGedFrame.h"
#endif

class TGComboBox;
class TGColorSelect;
class TGedMarkerSelect;
class TGFontTypeComboBox;
class TAttMarker;

class TAttMarkerEditor : public TGedFrame {

protected:
   TAttMarker          *fAttMarker;       // marker attribute object
   TGFontTypeComboBox  *fTypeCombo;       // font style combo box
   TGComboBox          *fSizeCombo;       // font size combo box
   TGColorSelect       *fColorSelect;     // color selection widget
   TGedMarkerSelect    *fMarkerSelect;    // marker selection widget

   static  TGComboBox *BuildMarkerSizeComboBox(TGFrame *parent, Int_t id);
   virtual void        ConnectSignals2Slots();

public:
   TAttMarkerEditor(const TGWindow *p, Int_t id,
                    Int_t width = 140, Int_t height = 30,
                    UInt_t options = kChildFrame,
                    Pixel_t back = GetDefaultFrameBackground());
   virtual ~TAttMarkerEditor();

   virtual void     SetModel(TVirtualPad *pad, TObject *obj, Int_t event);
   virtual void     DoMarkerColor(Pixel_t color);
   virtual void     DoMarkerSize(Int_t size);
   virtual void     DoMarkerStyle(Style_t style);

   ClassDef(TAttMarkerEditor,0)  // GUI for editing marker attributes
};

#endif
