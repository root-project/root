// @(#)root/gl:$Name:  $:$Id: TGLEditor.h,v 1.8 2004/11/23 21:42:55 brun Exp $
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

class TGCheckButton;
class TViewerOpenGL;
class TGLayoutHints;
class TGNumberEntry;
class TGLMatView;
class TGHSlider;
class TGButton;
class TGCanvas;
class TGLabel;

enum EApplyButtonIds {
   kTBcp,
   kTBcpm,
   kTBda,
   kTBa,
   kTBaf,
   kTBTop,
   kTBRight,
   kTBBottom,
   kTBLeft,
   kTBFront,            
   kTBa1
};

class TGLColorEditor : public TGCompositeFrame {
   friend class TGLMatView;
private:
   TViewerOpenGL *fViewer;
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
   TGButton      *fApplyFamily;
   Bool_t        fIsActive;
   Bool_t        fIsLight;   
   Float_t       fRGBA[17];

   Window_t      fGLWin;
   ULong_t       fCtx;

   TList fTrash;
public:
   TGLColorEditor(const TGWindow *parent, TViewerOpenGL *viewer);
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

   TViewerOpenGL  *fViewer;
   TList          fTrash;
   TGLayoutHints  *fL1, *fL2;
   TGNumberEntry  *fGeomData[kTot];
   TGButton       *fApplyButton;
   Bool_t         fIsActive;

   Double_t       fCenter[3];

public:
   TGLGeometryEditor(const TGWindow *parent, TViewerOpenGL *viewer);

   void SetCenter(const Double_t *center);
   void Disable();
   void DoButton();
   void GetObjectData(Double_t *shift, Double_t *scale);
   void ValueSet(Long_t unusedVal);

private:
   void CreateCenterControls();
   void CreateStretchControls();

   TGLGeometryEditor(const TGLGeometryEditor &);
   TGLGeometryEditor &operator = (const TGLGeometryEditor &);

   ClassDef(TGLGeometryEditor, 0)
};

class TGLSceneEditor : public TGCompositeFrame {
private:
   enum {
      kPlaneA,
      kPlaneB,
      kPlaneC,
      kPlaneD,
      kTot
   };

   TViewerOpenGL  *fViewer;
   TList          fTrash;
   TGLayoutHints  *fL1, *fL2;
   TGNumberEntry  *fGeomData[kTot];
   TGButton       *fApplyButton;
   TGCheckButton  *fClipActivate;
   TGCheckButton  *fAxesCheck;

public:
   TGLSceneEditor(const TGWindow *parent, TViewerOpenGL *viewer);   

   void GetPlaneEqn(Double_t *eqn);
   void DoButton();
   void ValueSet(Long_t unusedVal);

private:
   void CreateControls();

   TGLSceneEditor(const TGLSceneEditor &);
   TGLSceneEditor &operator = (const TGLSceneEditor &);

   ClassDef(TGLSceneEditor, 0);
};

class TGLLightEditor : public TGCompositeFrame {
private:
   enum EBuiltInLight {
      kTop,
      kRight,
      kBottom,
      kLeft,
      kFront,
      kTot      
   };
   
   TGButton       *fLights[kTot];
   TViewerOpenGL  *fViewer;
   TList          fTrash;
   
public:
   TGLLightEditor(const TGWindow *parent, TViewerOpenGL *viewer);
   
   void DoButton();
   
   ClassDef(TGLLightEditor, 0);
};

#endif
