// @(#)root/eve:$Id$
// Authors: Alja & Matevz Tadel 2008

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TEveTextEditor
#define ROOT_TEveTextEditor

#include "TGedFrame.h"

class TGComboBox;
class TGTextEntry;
class TGCheckButton;
class TEveGValuator;

class TEveText;

class TEveTextEditor : public TGedFrame
{
private:
   TEveTextEditor(const TEveTextEditor&);            // Not implemented
   TEveTextEditor& operator=(const TEveTextEditor&); // Not implemented

   TGComboBox* MakeLabeledCombo(const char* name);

protected:
   TEveText            *fM;     // Model object.

   TGTextEntry         *fText;
   TGComboBox          *fSize;
   TGComboBox          *fFile;
   TGComboBox          *fMode;
   TEveGValuator       *fExtrude;

   TGCheckButton       *fLighting;
   TGCheckButton       *fAutoLighting;

public:
   TEveTextEditor(const TGWindow* p=0, Int_t width=170, Int_t height=30,
                  UInt_t options=kChildFrame, Pixel_t back=GetDefaultFrameBackground());
   virtual ~TEveTextEditor() {}

   virtual void SetModel(TObject* obj);

   void DoText(const char*);

   void DoFontSize();
   void DoFontFile();
   void DoFontMode();

   void DoLighting();
   void DoAutoLighting();
   void DoExtrude();

   ClassDef(TEveTextEditor, 0); // GUI editor for TEveText.
};

#endif
