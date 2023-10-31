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
   TGLLightSetSubEditor(const TGLLightSetSubEditor&) = delete;
   TGLLightSetSubEditor& operator=(const TGLLightSetSubEditor&) = delete;

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
   ~TGLLightSetSubEditor() override {}

   void SetModel(TGLLightSet* m);

   void Changed(); //*SIGNAL*

   void DoButton();

   ClassDefOverride(TGLLightSetSubEditor, 0) // Sub-editor for TGLLightSet.
};


class TGLLightSetEditor : public TGedFrame
{
private:
   TGLLightSetEditor(const TGLLightSetEditor&) = delete;
   TGLLightSetEditor& operator=(const TGLLightSetEditor&) = delete;

protected:
   TGLLightSet          *fM;  // fModel dynamic-casted to TGLLightSetEditor
   TGLLightSetSubEditor *fSE;

public:
   TGLLightSetEditor(const TGWindow *p = nullptr, Int_t width=170, Int_t height=30, UInt_t options=kChildFrame, Pixel_t back=GetDefaultFrameBackground());
   ~TGLLightSetEditor() override;

   void SetModel(TObject* obj) override;

   ClassDefOverride(TGLLightSetEditor, 0); // Editor for TGLLightSet.
}; // endclass TGLLightSetEditor

#endif
