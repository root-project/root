// @(#)root/eve:$Id$
// Authors: Matevz Tadel & Alja Mrak-Tadel: 2006, 2007

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TEveGedEditor
#define ROOT_TEveGedEditor

#include "TGedEditor.h"

class TEveElement;

class TEveGedEditor : public TGedEditor
{
   TEveGedEditor(const TEveGedEditor&);            // Not implemented
   TEveGedEditor& operator=(const TEveGedEditor&); // Not implemented

protected:
   TEveElement   *fElement;    // Cached eve-element pointer.
   TObject       *fObject;     // Cached tobj pointer.

public:
   TEveGedEditor(TCanvas* canvas=0, Int_t width=250, Int_t height=400);
   virtual ~TEveGedEditor() {}

   TEveElement* GetEveElement() const;

   void DisplayElement(TEveElement* re);
   void DisplayObject(TObject* obj);

   virtual void SetModel(TVirtualPad* pad, TObject* obj, Int_t event);
   virtual void Update(TGedFrame* gframe=0);

   // virtual Bool_t HandleButton(Event_t *event);

   ClassDef(TEveGedEditor, 0); // Specialization of TGedEditor for proper update propagation to TEveManager.
};

#endif
