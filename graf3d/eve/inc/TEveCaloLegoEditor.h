// @(#)root/eve:$Id$
// Author: Matevz Tadel 2007

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TEveCaloLegoEditor
#define ROOT_TEveCaloLegoEditor

#include "TGedFrame.h"

class TGButton;
class TGCheckButton;
class TGNumberEntry;
class TGColorSelect;
class TGComboBox;
class TEveGValuator;

class TEveCaloLego;

class TEveCaloLegoEditor : public TGedFrame
{
private:
   TEveCaloLegoEditor(const TEveCaloLegoEditor&);            // Not implemented
   TEveCaloLegoEditor& operator=(const TEveCaloLegoEditor&); // Not implemented
   TGComboBox*  MakeLabeledCombo(const char* name, Int_t off);

protected:
   TEveCaloLego      *fM; // Model object.

   TGColorSelect*     fGridColor;

   TGColorSelect*     fFontColor;

   TEveGValuator     *fNZStep;
   TEveGValuator     *fBinWidth;

   TGComboBox        *fProjection;
   TGComboBox        *f2DMode;
   TGComboBox        *fBoxMode;

public:
   TEveCaloLegoEditor(const TGWindow* p=0, Int_t width=170, Int_t height=30,
         UInt_t options=kChildFrame, Pixel_t back=GetDefaultFrameBackground());
   virtual ~TEveCaloLegoEditor() {}

   virtual void SetModel(TObject* obj);

   // Declare callback/slot methods
   void DoFontColor(Pixel_t color);
   void DoGridColor(Pixel_t color);

   void DoNZStep();

   void DoBinWidth();

   void DoProjection();
   void Do2DMode();
   void DoBoxMode();

   ClassDef(TEveCaloLegoEditor, 0); // GUI editor for TEveCaloLego.
};

#endif
