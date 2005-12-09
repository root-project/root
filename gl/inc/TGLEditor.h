// @(#)root/gl:$Name:  $:$Id: TGLEditor.h,v 1.15 2005/11/23 10:47:52 brun Exp $
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
#ifndef ROOT_TGLViewer
#include "TGLViewer.h" // For viewer enums - to be reconsidered
#endif

class TGLStandalone;
class TGLSAViewer;
class TGCheckButton;
class TGLayoutHints;
class TGNumberEntry;
class TGLMatView;
class TGHSlider;
class TGButton;
class TGButtonGroup;
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
   kTBa1,
   kTBGuide
};

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGLColorEditor                                                       //
//                                                                      //
// GL Viewer shape color editor GUI component                           //
//////////////////////////////////////////////////////////////////////////

class TGLColorEditor : public TGCompositeFrame {
   friend class TGLMatView;
private:
   TGLSAViewer   *fViewer;
   TGLMatView    *fMatView;
   TGLayoutHints *fFrameLayout;

   enum ELightMode{kDiffuse, kAmbient, kSpecular, kEmission, kTot};
   ELightMode     fLMode;
   TGButton      *fLightTypes[kTot];

   TGHSlider     *fRedSlider;
   TGHSlider     *fGreenSlider;
   TGHSlider     *fBlueSlider;
   TGHSlider     *fAlphaSlider;
   TGHSlider     *fShineSlider;

   TGButton      *fApplyButton;
   TGButton      *fApplyFamily;
   Bool_t         fIsActive;
   Bool_t         fIsLight;   
   Float_t        fRGBA[17];

   Window_t       fGLWin;
   ULong_t        fCtx;

   TList fTrash;
public:
   TGLColorEditor(const TGWindow *parent, TGLSAViewer *viewer);
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

   ClassDef(TGLColorEditor, 0); // GL Viewer shape color editor GUI component
};

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGLGeometryEditor                                                    //
//                                                                      //
// GL Viewer shape geometry editor GUI component                        //
//////////////////////////////////////////////////////////////////////////

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

   TGLSAViewer   *fViewer;
   TList          fTrash;
   TGLayoutHints  *fL1, *fL2;
   TGNumberEntry  *fGeomData[kTot];
   TGButton       *fApplyButton;
   Bool_t         fIsActive;

public:
   TGLGeometryEditor(const TGWindow *parent, TGLSAViewer *viewer);

   void SetCenter(const Double_t *center);
   void SetScale(const Double_t *scale);
   void Disable();
   void DoButton();
   void GetObjectData(Double_t *shift, Double_t *scale);
   void ValueSet(Long_t unusedVal);

private:
   void CreateCenterControls();
   void CreateStretchControls();

   TGLGeometryEditor(const TGLGeometryEditor &);
   TGLGeometryEditor &operator = (const TGLGeometryEditor &);

   ClassDef(TGLGeometryEditor, 0); // GL Viewer shape geometry editor GUI component
};

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGLClipEditor                                                        //
//                                                                      //
// GL Viewer clipping shape editor GUI component                        //
//////////////////////////////////////////////////////////////////////////

class TGLClipEditor : public TGCompositeFrame {
private:

   TGLSAViewer         *fViewer;
   TList                fTrash;
   TGLayoutHints       *fL1, *fL2;
   TGButton            *fApplyButton;
   TGButtonGroup       *fTypeButtons;
   TGCompositeFrame    *fPlanePropFrame;
   TGNumberEntry       *fPlaneProp[4];
   TGCompositeFrame    *fBoxPropFrame;
   TGNumberEntry       *fBoxProp[6];
   TGCheckButton       *fEdit;
   TGLViewer::EClipType fCurrentClip; // Nasty - need to move
                                      // all common enums out somewhere else

public:
   TGLClipEditor(const TGWindow *parent, TGLSAViewer *viewer);   

   // Internal GUI event callbacks
   void ClipValueChanged(Long_t);
   void ClipTypeChanged(Int_t);
   void UpdateViewer();

   // External viewer interface
   void GetState(TGLViewer::EClipType type, Double_t data[6]) const;
   void SetState(TGLViewer::EClipType type, const Double_t data[6]);
   void GetCurrent(TGLViewer::EClipType & type, Bool_t & edit) const;
   void SetCurrent(TGLViewer::EClipType type);

   void HideParts();

private:
   void CreateControls();

   TGLClipEditor(const TGLClipEditor &);
   TGLClipEditor &operator = (const TGLClipEditor &);

   ClassDef(TGLClipEditor, 0); //GL Viewer clipping shape editor GUI component
};

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGLLightEditor                                                       //
//                                                                      //
// GL Viewer lighting editor GUI component                              //
//////////////////////////////////////////////////////////////////////////

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
   TGLSAViewer    *fViewer;
   TList           fTrash;
   
   TGLLightEditor(const TGLLightEditor &);
   TGLLightEditor &operator = (const TGLLightEditor &);

public:
   TGLLightEditor(const TGWindow *parent, TGLSAViewer *viewer);
   
   void DoButton();
   
   ClassDef(TGLLightEditor, 0); // GL Viewer lighting editor GUI component
};

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGLGuideEditor                                                       //
//                                                                      //
// GL Viewer guides editor GUI component                                //
//////////////////////////////////////////////////////////////////////////

class TGLGuideEditor : public TGCompositeFrame {
private:
   TGLSAViewer    * fViewer;
   TGButtonGroup  * fAxesContainer;
   TGGroupFrame   * fReferenceContainer;
   TGCheckButton  * fReferenceOn;
   TGNumberEntry  * fReferencePos[3];

   TGLayoutHints  * fL1;
   TGLayoutHints  * fL2;
   TList            fTrash;

   void UpdateReferencePos();

   TGLGuideEditor(const TGLGuideEditor &);
   TGLGuideEditor &operator = (const TGLGuideEditor &);

public:
   TGLGuideEditor(const TGWindow *parent, TGLSAViewer *viewer);

   void Update();
   void GetState(TGLViewer::EAxesType & axesType, Bool_t & referenceOn, Double_t referencePos[3]) const;
   void SetState(TGLViewer::EAxesType axesType, Bool_t referenceOn, const Double_t referencePos[3]);

   ClassDef(TGLGuideEditor, 0); // GL Viewer guides editor GUI component
};

#endif
