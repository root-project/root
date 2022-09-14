// @(#)root/eve:$Id$
// Authors: Matevz Tadel & Alja Mrak-Tadel: 2006, 2007

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TEvePointSetArrayEditor
#define ROOT_TEvePointSetArrayEditor

#include "TGedFrame.h"

class TGCheckButton;
class TGNumberEntry;
class TGColorSelect;

class TEveGValuator;
class TEveGDoubleValuator;

class TEvePointSetArray;

class TEvePointSetArrayEditor : public TGedFrame
{
   TEvePointSetArrayEditor(const TEvePointSetArrayEditor&);            // Not implemented
   TEvePointSetArrayEditor& operator=(const TEvePointSetArrayEditor&); // Not implemented

protected:
   TEvePointSetArray   *fM;       // Model object.

   TEveGDoubleValuator *fRange;   // Control for displayed range of the separating quantity.

public:
   TEvePointSetArrayEditor(const TGWindow *p = nullptr, Int_t width=170, Int_t height=30,
                           UInt_t options=kChildFrame, Pixel_t back=GetDefaultFrameBackground());
   ~TEvePointSetArrayEditor();

   virtual void SetModel(TObject* obj);

   void DoRange();

   ClassDef(TEvePointSetArrayEditor, 0); // Editor for TEvePointSetArray class.
};

#endif
