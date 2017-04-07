// @(#)root/gl:$Id$
// Author:  Matevz Tadel, Feb 2007

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGLLightSetEditor_H
#define ROOT_TGLLightSetEditor_H

#include <TGedFrame.h>

class TGButton;
class TGLLightSet;

class TGLLightSetSubEditor : public TGVerticalFrame
{
private:
   TGLLightSetSubEditor(const TGLLightSetSubEditor&);            // Not implemented
   TGLLightSetSubEditor& operator=(const TGLLightSetSubEditor&); // Not implemented

protected:
   TGLLightSet      *fM;

   TGGroupFrame     *fLightFrame;
   TGButton         *fTopLight;
   TGButton         *fRightLight;
   TGButton         *fBottomLight;
   TGButton         *fLeftLight;
   TGButton         *fFrontLight;

   TGButton         *fSpecularLight;

   TGButton* MakeLampButton(const char* name, Int_t wid, TGCompositeFrame* parent);

public:
   TGLLightSetSubEditor(const TGWindow* p);
   virtual ~TGLLightSetSubEditor() {}

   void SetModel(TGLLightSet* m);

   void Changed(); //*SIGNAL*

   void DoButton();

   ClassDef(TGLLightSetSubEditor, 0) // Sub-editor for TGLLightSet.
};


class TGLLightSetEditor : public TGedFrame
{
private:
   TGLLightSetEditor(const TGLLightSetEditor&);            // Not implemented
   TGLLightSetEditor& operator=(const TGLLightSetEditor&); // Not implemented

protected:
   TGLLightSet          *fM;  // fModel dynamic-casted to TGLLightSetEditor
   TGLLightSetSubEditor *fSE;

public:
   TGLLightSetEditor(const TGWindow* p=0, Int_t width=170, Int_t height=30, UInt_t options=kChildFrame, Pixel_t back=GetDefaultFrameBackground());
   virtual ~TGLLightSetEditor();

   virtual void SetModel(TObject* obj);

   ClassDef(TGLLightSetEditor, 0); // Editor for TGLLightSet.
}; // endclass TGLLightSetEditor

#endif
