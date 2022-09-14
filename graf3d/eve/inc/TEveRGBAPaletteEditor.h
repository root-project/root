// @(#)root/eve:$Id$
// Authors: Matevz Tadel & Alja Mrak-Tadel: 2006, 2007

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TEveRGBAPaletteEditor
#define ROOT_TEveRGBAPaletteEditor

#include "TGedFrame.h"

class TGCheckButton;
class TGColorSelect;
class TGComboBox;

class TEveRGBAPalette;
class TEveGValuator;
class TEveGDoubleValuator;

class TEveRGBAPaletteSubEditor : public TGVerticalFrame
{
private:
   TEveRGBAPaletteSubEditor(const TEveRGBAPaletteSubEditor&);            // Not implemented
   TEveRGBAPaletteSubEditor& operator=(const TEveRGBAPaletteSubEditor&); // Not implemented

protected:
   TEveRGBAPalette      *fM;

   TGComboBox           *fUnderflowAction;
   TGColorSelect        *fUnderColor;
   TGComboBox           *fOverflowAction;
   TGColorSelect        *fOverColor;

   TEveGDoubleValuator  *fMinMax;
   Double_t              fOldMin;
   Double_t              fOldMax;

   TGCheckButton        *fInterpolate;
   TGCheckButton        *fShowDefValue;
   TGColorSelect        *fDefaultColor;
   TGCheckButton        *fFixColorRange;

public:
   TEveRGBAPaletteSubEditor(const TGWindow* p);
   virtual ~TEveRGBAPaletteSubEditor() {}

   void SetModel(TEveRGBAPalette* p);

   void Changed(); //*SIGNAL*

   void DoMinMax();

   void DoInterpolate();
   void DoShowDefValue();
   void DoDefaultColor(Pixel_t color);
   void DoFixColorRange();
   void DoUnderColor(Pixel_t color);
   void DoOverColor(Pixel_t color);
   void DoUnderflowAction(Int_t mode);
   void DoOverflowAction(Int_t mode);

   ClassDef(TEveRGBAPaletteSubEditor, 0); // Sub-editor for TEveRGBAPalette class.
};


/******************************************************************************/
/******************************************************************************/

class TEveRGBAPaletteEditor : public TGedFrame
{
private:
   TEveRGBAPaletteEditor(const TEveRGBAPaletteEditor&);            // Not implemented
   TEveRGBAPaletteEditor& operator=(const TEveRGBAPaletteEditor&); // Not implemented

protected:
   TEveRGBAPalette           *fM;
   TEveRGBAPaletteSubEditor  *fSE;

public:
   TEveRGBAPaletteEditor(const TGWindow *p = nullptr, Int_t width=170, Int_t height=30, UInt_t options = kChildFrame, Pixel_t back=GetDefaultFrameBackground());
   virtual ~TEveRGBAPaletteEditor() {}

   virtual void SetModel(TObject* obj);

   ClassDef(TEveRGBAPaletteEditor, 0); // Editor for TEveRGBAPalette class.
};

#endif
