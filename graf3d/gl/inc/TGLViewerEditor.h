// @(#)root/gl:$Id$
// Author:  Alja Mrak-Tadel, Matevz Tadel, Timur Pocheptsov 08/03/2006

/*************************************************************************
 * Copyright (C) 1995-2006, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGLViewerEditor
#define ROOT_TGLViewerEditor

#include <memory>

#ifndef ROOT_TGedFrame
#include "TGedFrame.h"
#endif

#ifndef ROOT_TGLUtil
#include "TGLUtil.h"
#endif
 
class TGCheckButton;
class TGNumberEntry;
class TGButtonGroup;
class TGroupFrame;
class TGRadioButton;
class TGColorSelect;
class TGComboBox;
class TGButton;
class TGTextEntry;
class TGLViewer;
class TGTab;

class TGLLightSetSubEditor;
class TGLClipSetSubEditor;

class TGLViewerEditor : public TGedFrame
{
private:
   //Pointers to manipulate with tabs
   TGCompositeFrame *fGuidesFrame;
   TGCompositeFrame *fClipFrame;
   TGCompositeFrame *fStereoFrame;

   TGLLightSetSubEditor *fLightSet;

   TGColorSelect    *fClearColor;
   TGCheckButton    *fIgnoreSizesOnUpdate;
   TGCheckButton    *fResetCamerasOnUpdate;
   TGTextButton     *fUpdateScene;
   TGTextButton     *fCameraHome;

   TGNumberEntry    *fMaxSceneDrawTimeHQ;
   TGNumberEntry    *fMaxSceneDrawTimeLQ;

   TGNumberEntry    *fPointSizeScale;
   TGNumberEntry    *fLineWidthScale;
   TGCheckButton    *fPointSmooth;
   TGCheckButton    *fLineSmooth;
   TGNumberEntry    *fWFLineWidth;
   TGNumberEntry    *fOLLineWidth;

   //"Guides" tab's controls
   TGCheckButton    *fCameraCenterExt;
   TGTextButton     *fCaptureCenter;
   TGCheckButton    *fDrawCameraCenter;
   TGNumberEntry    *fCameraCenterX;
   TGNumberEntry    *fCameraCenterY;
   TGNumberEntry    *fCameraCenterZ;

   TGCheckButton*    fCaptureAnnotate;

   Int_t             fAxesType;
   TGButtonGroup    *fAxesContainer;
   TGRadioButton    *fAxesNone;
   TGRadioButton    *fAxesEdge;
   TGRadioButton    *fAxesOrigin;
   TGCheckButton    *fAxesDepthTest;

   TGGroupFrame     *fRefContainer;
   TGCheckButton    *fReferenceOn;
   TGNumberEntry    *fReferencePosX;
   TGNumberEntry    *fReferencePosY;
   TGNumberEntry    *fReferencePosZ;

   TGGroupFrame     *fCamContainer;
   TGComboBox*       fCamMode;
   TGCheckButton*    fCamOverlayOn;

   TGLClipSetSubEditor *fClipSet;

   //'Extras' tab.
   TGCheckButton    *fRotateSceneOn;
   TGNumberEntry    *fSceneRotDt;
   
   TGNumberEntry    *fARotDt,     *fARotWPhi;
   TGNumberEntry    *fARotATheta, *fARotWTheta;
   TGNumberEntry    *fARotADolly, *fARotWDolly;

   TGTextEntry      *fASavImageGUIBaseName;
   TGButtonGroup    *fASavImageGUIOutMode;

   TGNumberEntry    *fStereoZeroParallax;
   TGNumberEntry    *fStereoEyeOffsetFac;
   TGNumberEntry    *fStereoFrustumAsymFac;

   // Model
   TGLViewer        *fViewer;
   Bool_t	     fIsInPad;

   void ConnectSignals2Slots();

   TGLViewerEditor(const TGLViewerEditor &);
   TGLViewerEditor &operator = (const TGLViewerEditor &);

   void CreateStyleTab();
   void CreateGuidesTab();
   void CreateClippingTab();
   void CreateExtrasTab();

   void UpdateReferencePosState();

public:
   TGLViewerEditor(const TGWindow *p=0, Int_t width=140, Int_t height=30,
                   UInt_t options=kChildFrame, Pixel_t back=GetDefaultFrameBackground());
   ~TGLViewerEditor();

   virtual void ViewerRedraw();

   virtual void SetModel(TObject* obj);

   void SetGuides();
   void DoClearColor(Pixel_t color);
   void DoIgnoreSizesOnUpdate();
   void DoResetCamerasOnUpdate();
   void DoUpdateScene();
   void DoCameraHome();
   void UpdateMaxDrawTimes();
   void UpdatePointLineStuff();
   void DoCameraCenterExt();
   void DoCaptureCenter();
   void DoAnnotation();
   void DoDrawCameraCenter();
   void UpdateCameraCenter();
   // Axis manipulation
   void UpdateViewerAxes(Int_t id);
   void UpdateViewerReference();
   void DoCameraOverlay();
   // Extras
   void SetRotatorMode();
   void UpdateRotator();
   void DoRotatorStart();
   void DoRotatorStop();
   void DoASavImageGUIBaseName(const char* t);
   void DoASavImageGUIOutMode(Int_t m);
   void DoASavImageStart();
   void DoASavImageStop();
   void UpdateStereo();

   void DetachFromPad(){fIsInPad = kFALSE;}

   static TGNumberEntry* MakeLabeledNEntry(TGCompositeFrame* p, const char* name,
                                           Int_t labelw, Int_t nd=7, Int_t s=5);

   ClassDef(TGLViewerEditor, 0); //GUI for editing TGLViewer attributes
};

#endif
