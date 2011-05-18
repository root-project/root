// @(#)root/eve:$Id$
// Author: Matevz Tadel 2007

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TEveArrowEditor
#define ROOT_TEveArrowEditor

#include "TGedFrame.h"

class TEveGValuator;
class TEveArrow;
class TEveGTriVecValuator;

class TEveArrowEditor : public TGedFrame
{
private:
   TEveArrowEditor(const TEveArrowEditor&);            // Not implemented
   TEveArrowEditor& operator=(const TEveArrowEditor&); // Not implemented

protected:
   TEveArrow            *fM; // Model object.

   TEveGValuator        *fTubeR;
   TEveGValuator        *fConeR;
   TEveGValuator        *fConeL;

   TEveGTriVecValuator  *fOrigin;
   TEveGTriVecValuator  *fVector;

public:
   TEveArrowEditor(const TGWindow* p=0, Int_t width=170, Int_t height=30,
         UInt_t options=kChildFrame, Pixel_t back=GetDefaultFrameBackground());
   virtual ~TEveArrowEditor() {}

   virtual void SetModel(TObject* obj);

   void DoTubeR();
   void DoConeR();
   void DoConeL();
   void DoVertex();

   ClassDef(TEveArrowEditor, 0); // GUI editor for TEveArrow.
};

#endif
