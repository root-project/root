// @(#)root/gl:$Name:  $:$Id: TGLColorEditor.h,v 1.3 2004/09/29 06:55:13 brun Exp $
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
class TGNumberEntry;
class TGLMatView;
class TGHSlider;
class TGButton;
class TGCanvas;
class TGLabel;

enum EApplyButtonIds {
   kTBa,
   kTBa1
};

class TGLColorEditor : public TGCompositeFrame {
   friend class TGLMatView;
private:
   TGLMatView    *fMatView;
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
   TGLColorEditor(const TGWindow *parent, TGWindow *main);
   ~TGLColorEditor();
   void SetRGBA(const Float_t *rgba);
   const Float_t *GetRGBA()const
   {
      return fRGBA;
   }
   //slots
   void DoSlider(Int_t val);
   void DoButton();
   void Disable();

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
   TGLColorEditor(const TGLColorEditor &);
   TGLColorEditor & operator = (const TGLColorEditor &);

   ClassDef(TGLColorEditor, 0)
};

class TGLGeometryEditor : public TGCompositeFrame {
private:
   enum {
      kCenterX, 
      kCenterY, 
      kCenterZ, 
      kScaleX, 
      kScaleY, 
      kScaleZ, 
      kTot
   };

   TList         fTrash;
   TGLayoutHints *fFrameLayout;
   TGNumberEntry *fGeomData[kTot];
   TGButton      *fApplyButton;

   Bool_t fIsActive;

public:
   TGLGeometryEditor(const TGWindow *parent, TGWindow *main);

   void SetCenter(const Double_t *center);
   void Disable();
   void DoButton();
   void GetNewData(Double_t *center, Double_t *scale);
   void ValueSet(Long_t val);

private:
   void CreateCenterFrame();
   void CreateScaleFrame();

   TGLGeometryEditor(const TGLGeometryEditor &);
   TGLGeometryEditor &operator = (const TGLGeometryEditor &);

   ClassDef(TGLGeometryEditor, 0)
};

#endif
