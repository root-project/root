// @(#)root/gl:$Name:  $:$Id: TGLEditor.h,v 1.2 2004/09/14 15:32:47 rdm Exp $
// Author:  Timur Pocheptsov  03/08/2004

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGLEditor
#define ROOT_TGLEditor

#ifndef ROOT_RQ_OBJECT
#include "RQ_OBJECT.h"
#endif
#ifndef ROOT_TGFrame
#include "TGFrame.h"
#endif
#ifndef ROOT_TList
#include "TList.h"
#endif


class TGLayoutHints;
class TGLMatView;
class TGHSlider;
class TGButton;
class TGCanvas;
class TGLabel;

class TGLEditor : public TGCompositeFrame {
   friend class TGLMatView;
private:
   TGLMatView    *fMatView;
   TGGroupFrame  *fPartFrame;
   TGLayoutHints *fFrameLayout;

   enum ELightMode{kDiffuse, kAmbient, kSpecular, kEmission, kTot};
   ELightMode    fLMode;
   TGButton      *fLightTypes[kTot];

   TGHSlider     *fRedSlider;
   TGHSlider     *fGreenSlider;
   TGHSlider     *fBlueSlider;
   TGHSlider     *fAlphaSlider;
   TGHSlider     *fShineSlider;

   TGButton      *fApplyButton;
   Bool_t        fIsActive;
   Bool_t        fIsLight;   
   Float_t       fRGBA[17];

   Window_t      fGLWin;
   ULong_t       fCtx;

   TList fTrash;
public:
   TGLEditor(const TGWindow *parent, TGWindow *main);
   ~TGLEditor();
   void SetRGBA(const Float_t *rgba);
   const Float_t *GetRGBA()const
   {
      return fRGBA;
   }
   //slots
   void DoSlider(Int_t val);
   void DoButton();
   void Stop();

private:
   void CreateRadioButtons();
   void CreateSliders();
   void SetSlidersPos();
   Bool_t HandleContainerNotify(Event_t *event);
   Bool_t HandleContainerExpose(Event_t *event);
   void DrawSphere()const;
   void SwapBuffers()const;
   void MakeCurrent()const;
   //Non-copyable class
   TGLEditor(const TGLEditor &);
   TGLEditor & operator = (const TGLEditor &);

   ClassDef(TGLEditor, 0)
};

#endif
