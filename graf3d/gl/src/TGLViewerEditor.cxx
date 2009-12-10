#include <cstring>

#include "TGedEditor.h"
#include "TGNumberEntry.h"
#include "TGButtonGroup.h"
#include "TGColorSelect.h"
#include "TVirtualGL.h"
#include "TG3DLine.h"
#include "TGButton.h"
#include "TColor.h"
#include "TString.h"
#include "TGLabel.h"
#include "TClass.h"
#include "TGTab.h"
#include "TGComboBox.h"

#include "TGLViewerEditor.h"
#include "TGLViewer.h"
#include "TGLLightSetEditor.h"
#include "TGLClipSetEditor.h"
#include "TGLUtil.h"
#include "TGLCameraOverlay.h"

//______________________________________________________________________________
//
// GUI editor for TGLViewer.

ClassImp(TGLViewerEditor);

//______________________________________________________________________________
TGLViewerEditor::TGLViewerEditor(const TGWindow *p,  Int_t width, Int_t height, UInt_t options, Pixel_t back) :
   TGedFrame(p,  width, height, options | kVerticalFrame, back),
   fGuidesFrame(0),
   fClipFrame(0),
   fClearColor(0),
   fIgnoreSizesOnUpdate(0),
   fResetCamerasOnUpdate(0),
   fUpdateScene(0),
   fCameraHome(0),
   fMaxSceneDrawTimeHQ(0),
   fMaxSceneDrawTimeLQ(0),
   fPointSizeScale(0),  fLineWidthScale(0),
   fPointSmooth(0),     fLineSmooth(0),
   fWFLineWidth(0),     fOLLineWidth(0),

   fCameraCenterExt(0),
   fCaptureCenter(0),
   fCameraCenterX(0),
   fCameraCenterY(0),
   fCameraCenterZ(0),
   fCaptureAnnotate(),
   fAxesType(0),
   fAxesContainer(0),
   fAxesNone(0),
   fAxesEdge(0),
   fAxesOrigin(0),
   fAxesDepthTest(0),
   fRefContainer(0),
   fReferenceOn(0),
   fReferencePosX(0),
   fReferencePosY(0),
   fReferencePosZ(0),
   fCamContainer(0),
   fCamMode(0),
   fCamOverlayOn(0),
   fClipSet(0),
   fStereoZeroParallax(0),
   fStereoEyeOffsetFac(0),
   fStereoFrustumAsymFac(0),
   fViewer(0),
   fIsInPad(kTRUE)
{
  //  Constructor.

   CreateStyleTab();
   CreateGuidesTab();
   CreateClippingTab();
   CreateStereoTab();
}

//______________________________________________________________________________

TGLViewerEditor::~TGLViewerEditor()
{
   // Destructor.

}

//______________________________________________________________________________
void TGLViewerEditor::ConnectSignals2Slots()
{
   // Connect signals to slots.

   fClearColor->Connect("ColorSelected(Pixel_t)", "TGLViewerEditor", this, "DoClearColor(Pixel_t)");
   fIgnoreSizesOnUpdate->Connect("Toggled(Bool_t)", "TGLViewerEditor", this, "DoIgnoreSizesOnUpdate()");
   fResetCamerasOnUpdate->Connect("Toggled(Bool_t)", "TGLViewerEditor", this, "DoResetCamerasOnUpdate()");
   fUpdateScene->Connect("Pressed()", "TGLViewerEditor", this, "DoUpdateScene()");
   fCameraHome->Connect("Pressed()", "TGLViewerEditor", this, "DoCameraHome()");
   fMaxSceneDrawTimeHQ->Connect("ValueSet(Long_t)", "TGLViewerEditor", this, "UpdateMaxDrawTimes()");
   fMaxSceneDrawTimeLQ->Connect("ValueSet(Long_t)", "TGLViewerEditor", this, "UpdateMaxDrawTimes()");
   fPointSizeScale->Connect("ValueSet(Long_t)", "TGLViewerEditor", this, "UpdatePointLineStuff()");
   fLineWidthScale->Connect("ValueSet(Long_t)", "TGLViewerEditor", this, "UpdatePointLineStuff()");
   fPointSmooth->Connect("Clicked()", "TGLViewerEditor", this, "UpdatePointLineStuff()");
   fLineSmooth ->Connect("Clicked()", "TGLViewerEditor", this, "UpdatePointLineStuff()");
   fWFLineWidth->Connect("ValueSet(Long_t)", "TGLViewerEditor", this, "UpdatePointLineStuff()");
   fOLLineWidth->Connect("ValueSet(Long_t)", "TGLViewerEditor", this, "UpdatePointLineStuff()");

   fCameraCenterExt->Connect("Clicked()", "TGLViewerEditor", this, "DoCameraCenterExt()");
   fCaptureCenter->Connect("Clicked()", "TGLViewerEditor", this, "DoCaptureCenter()");
   fDrawCameraCenter->Connect("Clicked()", "TGLViewerEditor", this, "DoDrawCameraCenter()");
   fCameraCenterX->Connect("ValueSet(Long_t)", "TGLViewerEditor", this, "UpdateCameraCenter()");
   fCameraCenterY->Connect("ValueSet(Long_t)", "TGLViewerEditor", this, "UpdateCameraCenter()");
   fCameraCenterZ->Connect("ValueSet(Long_t)", "TGLViewerEditor", this, "UpdateCameraCenter()");

   fCaptureAnnotate->Connect("Clicked()", "TGLViewerEditor", this, "DoAnnotation()");

   fAxesContainer->Connect("Clicked(Int_t)", "TGLViewerEditor", this, "UpdateViewerAxes(Int_t)");

   fReferenceOn->Connect("Clicked()", "TGLViewerEditor", this, "UpdateViewerReference()");
   fReferencePosX->Connect("ValueSet(Long_t)", "TGLViewerEditor", this, "UpdateViewerReference()");
   fReferencePosY->Connect("ValueSet(Long_t)", "TGLViewerEditor", this, "UpdateViewerReference()");
   fReferencePosZ->Connect("ValueSet(Long_t)", "TGLViewerEditor", this, "UpdateViewerReference()");

   fCamMode->Connect("Selected(Int_t)", "TGLViewerEditor", this, "DoCameraOverlay()");
   fCamOverlayOn->Connect("Clicked()", "TGLViewerEditor", this, "DoCameraOverlay()");

   fStereoZeroParallax  ->Connect("ValueSet(Long_t)", "TGLViewerEditor", this, "UpdateStereo()");
   fStereoEyeOffsetFac  ->Connect("ValueSet(Long_t)", "TGLViewerEditor", this, "UpdateStereo()");
   fStereoFrustumAsymFac->Connect("ValueSet(Long_t)", "TGLViewerEditor", this, "UpdateStereo()");
   fStereoZeroParallax  ->Connect("ValueChanged(Long_t)", "TGLViewerEditor", this, "UpdateStereo()");
   fStereoEyeOffsetFac  ->Connect("ValueChanged(Long_t)", "TGLViewerEditor", this, "UpdateStereo()");
   fStereoFrustumAsymFac->Connect("ValueChanged(Long_t)", "TGLViewerEditor", this, "UpdateStereo()");

   fInit = kFALSE;
}

//______________________________________________________________________________
void TGLViewerEditor::ViewerRedraw()
{
   // Initiate redraw of the viewer.

   if (gGLManager && fIsInPad)
      gGLManager->MarkForDirectCopy(fViewer->GetDev(), kTRUE);
   fViewer->RequestDraw();
}

//______________________________________________________________________________
void TGLViewerEditor::SetModel(TObject* obj)
{
   // Sets model or disables/hides viewer.

   fViewer = 0;

   fViewer = static_cast<TGLViewer *>(obj);
   fIsInPad = (fViewer->GetDev() != -1);

   SetGuides();

   if (fInit)
      ConnectSignals2Slots();

   fLightSet->SetModel(fViewer->GetLightSet());
   fClipSet->SetModel(fViewer->GetClipSet());

   // style tab
   fClearColor->SetColor(TColor::Number2Pixel(fViewer->RnrCtx().ColorSet().Background().GetColorIndex()), kFALSE);
   fClearColor->Enable(!fViewer->IsUsingDefaultColorSet());
   fIgnoreSizesOnUpdate->SetState(fViewer->GetIgnoreSizesOnUpdate() ? kButtonDown : kButtonUp);
   fResetCamerasOnUpdate->SetState(fViewer->GetResetCamerasOnUpdate() ? kButtonDown : kButtonUp);
   fMaxSceneDrawTimeHQ->SetNumber(fViewer->GetMaxSceneDrawTimeHQ());
   fMaxSceneDrawTimeLQ->SetNumber(fViewer->GetMaxSceneDrawTimeLQ());
   fPointSizeScale->SetNumber(fViewer->GetPointScale());
   fLineWidthScale->SetNumber(fViewer->GetLineScale ());
   fPointSmooth->SetState(fViewer->GetSmoothPoints() ? kButtonDown : kButtonUp);
   fLineSmooth ->SetState(fViewer->GetSmoothLines () ? kButtonDown : kButtonUp);
   fWFLineWidth->SetNumber(fViewer->WFLineW());
   fOLLineWidth->SetNumber(fViewer->OLLineW());
   //camera look at
   TGLCamera & cam = fViewer->CurrentCamera();
   fCameraCenterExt->SetDown(cam.GetExternalCenter());
   fDrawCameraCenter->SetDown(fViewer->GetDrawCameraCenter());
   Double_t* la = cam.GetCenterVec();
   fCameraCenterX->SetNumber(la[0]);
   fCameraCenterY->SetNumber(la[1]);
   fCameraCenterZ->SetNumber(la[2]);
   fCameraCenterX->SetState(fCameraCenterExt->IsDown());
   fCameraCenterY->SetState(fCameraCenterExt->IsDown());
   fCameraCenterZ->SetState(fCameraCenterExt->IsDown());

   // push action
   fCaptureCenter->SetTextColor((fViewer->GetPushAction() == TGLViewer::kPushCamCenter) ? 0xa03060 : 0x000000);
   fCaptureAnnotate->SetDown( (fViewer->GetPushAction() == TGLViewer::kPushAnnotate), kFALSE);

   if (fViewer->GetStereo())
   {
      fStereoZeroParallax  ->SetNumber(fViewer->GetStereoZeroParallax());
      fStereoEyeOffsetFac  ->SetNumber(fViewer->GetStereoEyeOffsetFac());
      fStereoFrustumAsymFac->SetNumber(fViewer->GetStereoFrustumAsymFac());
      fStereoFrame->MapWindow();
   }
   else
   {
      fStereoFrame->UnmapWindow();
   }
}

//______________________________________________________________________________
void TGLViewerEditor::DoClearColor(Pixel_t color)
{
   // Clear-color was changed.

   fViewer->RnrCtx().ColorSet().Background().SetColor(Color_t(TColor::GetColor(color)));
   ViewerRedraw();
}

//______________________________________________________________________________
void TGLViewerEditor::DoIgnoreSizesOnUpdate()
{
   // ResetCamerasOnUpdate was toggled.

   fViewer->SetIgnoreSizesOnUpdate(fIgnoreSizesOnUpdate->IsOn());
   if (fIgnoreSizesOnUpdate->IsOn())
      fViewer->UpdateScene();
}

//______________________________________________________________________________
void TGLViewerEditor::DoResetCamerasOnUpdate()
{
   // ResetCamerasOnUpdate was toggled.

   fViewer->SetResetCamerasOnUpdate(fResetCamerasOnUpdate->IsOn());
}

//______________________________________________________________________________
void TGLViewerEditor::DoUpdateScene()
{
   // UpdateScene was clicked.

   fViewer->UpdateScene();
}

//______________________________________________________________________________
void TGLViewerEditor::DoCameraHome()
{
   // CameraHome was clicked.

   fViewer->ResetCurrentCamera();
   ViewerRedraw();
}

//______________________________________________________________________________
void TGLViewerEditor::UpdateMaxDrawTimes()
{
   // Slot for fMaxSceneDrawTimeHQ and fMaxSceneDrawTimeLQ.

   fViewer->SetMaxSceneDrawTimeHQ(fMaxSceneDrawTimeHQ->GetNumber());
   fViewer->SetMaxSceneDrawTimeLQ(fMaxSceneDrawTimeLQ->GetNumber());
}

//______________________________________________________________________________
void TGLViewerEditor::UpdatePointLineStuff()
{
   // Slot for point-sizes and line-widths.

   fViewer->SetPointScale(fPointSizeScale->GetNumber());
   fViewer->SetLineScale (fLineWidthScale->GetNumber());
   fViewer->SetSmoothPoints(fPointSmooth->IsDown());
   fViewer->SetSmoothLines (fLineSmooth->IsDown());
   fViewer->SetWFLineW(fWFLineWidth->GetNumber());
   fViewer->SetOLLineW(fOLLineWidth->GetNumber());
   ViewerRedraw();
}

//______________________________________________________________________________
void TGLViewerEditor::DoCameraOverlay()
{
   // Update viewer with GUI state.

   TGLCameraOverlay* co = fViewer->GetCameraOverlay();

   if (fViewer->CurrentCamera().IsPerspective())
   {
      co->SetShowPerspective(fCamOverlayOn->IsDown());
      co->SetPerspectiveMode((TGLCameraOverlay::EMode)fCamMode->GetSelected());
   }
   else
   {
      co->SetShowOrthographic(fCamOverlayOn->IsDown());
      co->SetOrthographicMode((TGLCameraOverlay::EMode)fCamMode->GetSelected());
   }
   ViewerRedraw();
}

//______________________________________________________________________________
void TGLViewerEditor::DoCameraCenterExt()
{
   // Set external camera center.

   TGLCamera& cam = fViewer->CurrentCamera();
   cam.SetExternalCenter(fCameraCenterExt->GetState());

   fCameraCenterX->SetState(fCameraCenterExt->IsDown());
   fCameraCenterY->SetState(fCameraCenterExt->IsDown());
   fCameraCenterZ->SetState(fCameraCenterExt->IsDown());

   ViewerRedraw();
}

//______________________________________________________________________________
void TGLViewerEditor::DoCaptureCenter()
{
   // Capture camera-center via picking.

   fViewer->PickCameraCenter();
   ViewerRedraw();
}

//______________________________________________________________________________
void TGLViewerEditor::DoDrawCameraCenter()
{
   // Draw camera center.

   fViewer->SetDrawCameraCenter(fDrawCameraCenter->IsDown());
   ViewerRedraw();
}

//______________________________________________________________________________
void TGLViewerEditor::UpdateCameraCenter()
{
   // Update current camera with GUI state.

   TGLCamera& cam = fViewer->CurrentCamera();
   cam.SetCenterVec(fCameraCenterX->GetNumber(), fCameraCenterY->GetNumber(), fCameraCenterZ->GetNumber());
   ViewerRedraw();
}

//______________________________________________________________________________
void TGLViewerEditor::DoAnnotation()
{
   // Create annotation via picking.

   fViewer->PickAnnotate();
}

//______________________________________________________________________________
void TGLViewerEditor::UpdateViewerAxes(Int_t id)
{
   // Update viewer with GUI state.

   if(id < 4)
   {
      fAxesType = id -1;
      for (Int_t i = 1; i < 4; i++) {
         TGButton * button = fAxesContainer->GetButton(i);
         if (i == id)
            button->SetDown(kTRUE);
         else
            button->SetDown(kFALSE);
      }
   }
   Bool_t axdt = fAxesContainer->GetButton(4)->IsDown();
   const Double_t refPos[] = {fReferencePosX->GetNumber(), fReferencePosY->GetNumber(), fReferencePosZ->GetNumber()};
   fViewer->SetGuideState(fAxesType, axdt, fReferenceOn->IsDown(), refPos);
   UpdateReferencePosState();
}

//______________________________________________________________________________
void TGLViewerEditor::UpdateViewerReference()
{
   // Update viewer with GUI state.

   const Double_t refPos[] = {fReferencePosX->GetNumber(), fReferencePosY->GetNumber(), fReferencePosZ->GetNumber()};
   fViewer->SetGuideState(fAxesType,  fAxesContainer->GetButton(4)->IsDown(), fReferenceOn->IsDown(), refPos);
   UpdateReferencePosState();
}

//______________________________________________________________________________
TGNumberEntry* TGLViewerEditor::MakeLabeledNEntry(TGCompositeFrame* p, const char* name,
                                                  Int_t labelw,Int_t nd, Int_t style)
{
   // Helper function to create fixed width TGLabel and TGNumberEntry in same row.

   TGHorizontalFrame *rfr   = new TGHorizontalFrame(p);
   TGHorizontalFrame *labfr = new TGHorizontalFrame(rfr, labelw, 20, kFixedSize);
   TGLabel           *lab   = new TGLabel(labfr, name);
   labfr->AddFrame(lab, new TGLayoutHints(kLHintsLeft | kLHintsBottom, 0, 0, 0) );
   rfr->AddFrame(labfr, new TGLayoutHints(kLHintsLeft | kLHintsBottom, 0, 0, 0));

   TGNumberEntry* ne = new TGNumberEntry(rfr, 0.0f, nd, -1, (TGNumberFormat::EStyle)style);
   rfr->AddFrame(ne, new TGLayoutHints(kLHintsLeft | kLHintsExpandX | kLHintsBottom, 2, 0, 0));

   p->AddFrame(rfr, new TGLayoutHints(kLHintsLeft, 0, 0, 1, 0));
   return ne;
}

//______________________________________________________________________________
void TGLViewerEditor::CreateStyleTab()
{
   // Creates "Style" tab.

   MakeTitle("Update behaviour");
   fIgnoreSizesOnUpdate  = new TGCheckButton(this, "Ignore sizes");
   fIgnoreSizesOnUpdate->SetToolTipText("Ignore bounding-box sizes on scene update");
   AddFrame(fIgnoreSizesOnUpdate, new TGLayoutHints(kLHintsLeft, 4, 1, 1, 1));
   fResetCamerasOnUpdate = new TGCheckButton(this, "Reset on update");
   fResetCamerasOnUpdate->SetToolTipText("Reset camera on scene update");
   AddFrame(fResetCamerasOnUpdate, new TGLayoutHints(kLHintsLeft, 4, 1, 1, 1));

   TGCompositeFrame* af = this;
   fUpdateScene = new TGTextButton(af, "Update Scene", 130);
   af->AddFrame(fUpdateScene, new TGLayoutHints(kLHintsLeft | kLHintsExpandX, 1, 1, 8, 1));
   fCameraHome = new TGTextButton(af, "Camera Home", 130);
   af->AddFrame(fCameraHome, new TGLayoutHints(kLHintsLeft | kLHintsExpandX, 1, 1, 1, 3));
   fMaxSceneDrawTimeHQ = MakeLabeledNEntry(af, "Max HQ draw time:", 120, 6, TGNumberFormat::kNESInteger);
   fMaxSceneDrawTimeHQ->SetLimits(TGNumberFormat::kNELLimitMin, 0, 1e6);
   fMaxSceneDrawTimeHQ->GetNumberEntry()->SetToolTipText("Maximum time spent in scene drawing\nin high-quality mode [ms].");
   fMaxSceneDrawTimeLQ = MakeLabeledNEntry(af, "Max LQ draw time:", 120, 6, TGNumberFormat::kNESInteger);
   fMaxSceneDrawTimeLQ->SetLimits(TGNumberFormat::kNELLimitMin, 0, 1e6);
   fMaxSceneDrawTimeLQ->GetNumberEntry()->SetToolTipText("Maximum time spent in scene drawing\nin low-quality mode (during rotation etc).");

   TGHorizontalFrame* hf = new TGHorizontalFrame(this);
   TGLabel* lab = new TGLabel(hf, "Clear Color");
   hf->AddFrame(lab, new TGLayoutHints(kLHintsLeft|kLHintsBottom, 1, 4, 8, 3));
   fClearColor = new TGColorSelect(hf, 0, -1);
   hf->AddFrame(fClearColor, new TGLayoutHints(kLHintsLeft, 1, 1, 8, 1));
   AddFrame(hf, new TGLayoutHints(kLHintsLeft, 2, 1, 1, 1));

   // LightSet
   fLightSet = new TGLLightSetSubEditor(this);
   fLightSet->Connect("Changed()", "TGLViewerEditor", this, "ViewerRedraw()");
   AddFrame(fLightSet, new TGLayoutHints(kLHintsTop | kLHintsExpandX, 2, 0, 0, 0));

   // Point-sizes / line-widths.
   hf = new TGHorizontalFrame(af);
   fPointSizeScale = MakeLabeledNEntry(hf, "Point-size scale:", 116, 4, TGNumberFormat::kNESRealOne);
   fPointSizeScale->SetLimits(TGNumberFormat::kNELLimitMinMax, 0.1, 16);
   fPointSmooth = new TGCheckButton(hf);
   fPointSmooth->SetToolTipText("Use smooth points.");
   hf->AddFrame(fPointSmooth, new TGLayoutHints(kLHintsNormal, 3, 0, 3, 0));
   af->AddFrame(hf);
   hf = new TGHorizontalFrame(af);
   fLineWidthScale = MakeLabeledNEntry(hf, "Line-width scale:", 116, 4, TGNumberFormat::kNESRealOne);
   fLineWidthScale->SetLimits(TGNumberFormat::kNELLimitMinMax, 0.1, 16);
   fLineSmooth = new TGCheckButton(hf);
   fLineSmooth->SetToolTipText("Use smooth lines.");
   hf->AddFrame(fLineSmooth, new TGLayoutHints(kLHintsNormal, 3, 0, 3, 0));
   af->AddFrame(hf);
   fWFLineWidth = MakeLabeledNEntry(af, "Wireframe line-width:", 116, 4, TGNumberFormat::kNESRealOne);
   fWFLineWidth->SetLimits(TGNumberFormat::kNELLimitMinMax, 0.1, 16);
   fOLLineWidth = MakeLabeledNEntry(af, "Outline line-width:", 116, 4, TGNumberFormat::kNESRealOne);
   fOLLineWidth->SetLimits(TGNumberFormat::kNELLimitMinMax, 0.1, 16);
}

//______________________________________________________________________________
void TGLViewerEditor::CreateGuidesTab()
{
   // Create "Guides" tab.
   fGuidesFrame = CreateEditorTabSubFrame("Guides");

   // external camera look at point
   TGGroupFrame* grf = new TGGroupFrame(fGuidesFrame, "Camera center:", kVerticalFrame);
   fDrawCameraCenter = new TGCheckButton(grf, "Show", 50);
   grf->AddFrame(fDrawCameraCenter, new TGLayoutHints(kLHintsTop | kLHintsLeft, 0, 0, 1, 1));
   fCameraCenterExt = new TGCheckButton(grf, "External", 50);
   grf->AddFrame(fCameraCenterExt, new TGLayoutHints(kLHintsLeft, 0, 0, 1, 0));
   fGuidesFrame->AddFrame(grf, new TGLayoutHints(kLHintsTop| kLHintsLeft | kLHintsExpandX, 2, 3, 3, 0));
   Int_t labw = 20;
   fCameraCenterX = MakeLabeledNEntry(grf, "X:", labw, 8, TGNumberFormat::kNESRealThree);
   fCameraCenterY = MakeLabeledNEntry(grf, "Y:", labw, 8, TGNumberFormat::kNESRealThree);
   fCameraCenterZ = MakeLabeledNEntry(grf, "Z:", labw, 8, TGNumberFormat::kNESRealThree);
   fCaptureCenter = new TGTextButton(grf, " Pick center ");
   grf->AddFrame(fCaptureCenter, new TGLayoutHints(kLHintsNormal, labw + 2, 0, 2, 0));

   // annotate
   TGGroupFrame* annf  = new TGGroupFrame(fGuidesFrame, "Annotation");
   fGuidesFrame->AddFrame(annf, new TGLayoutHints(kLHintsTop | kLHintsCenterX | kLHintsExpandX, 2, 3, 0, 0));
   fCaptureAnnotate = new TGCheckButton(annf, "Pick annotation");
   annf->AddFrame(fCaptureAnnotate, new TGLayoutHints(kLHintsTop | kLHintsCenterX | kLHintsExpandX));

   // reference container
   fRefContainer = new TGGroupFrame(fGuidesFrame, "Reference marker");
   fGuidesFrame->AddFrame(fRefContainer, new TGLayoutHints(kLHintsTop | kLHintsCenterX | kLHintsExpandX, 2, 3, 0, 0));
   fReferenceOn = new TGCheckButton(fRefContainer, "Show");
   fRefContainer->AddFrame(fReferenceOn, new TGLayoutHints(kLHintsTop | kLHintsCenterX | kLHintsExpandX));
   fReferencePosX = MakeLabeledNEntry(fRefContainer, "X:", labw, 8, TGNumberFormat::kNESRealThree );
   fReferencePosY = MakeLabeledNEntry(fRefContainer, "Y:", labw, 8, TGNumberFormat::kNESRealThree );
   fReferencePosZ = MakeLabeledNEntry(fRefContainer, "Z:", labw, 8, TGNumberFormat::kNESRealThree );

   // axes
   fAxesContainer = new TGButtonGroup(fGuidesFrame, "Axes");
   fAxesNone = new TGRadioButton(fAxesContainer, "None", 1);
   fAxesEdge = new TGRadioButton(fAxesContainer, "Edge", 2);
   fAxesOrigin = new TGRadioButton(fAxesContainer, "Origin", 3);
   fAxesDepthTest = new TGCheckButton(fAxesContainer, "DepthTest",4);
   fGuidesFrame->AddFrame(fAxesContainer, new TGLayoutHints(kLHintsTop | kLHintsCenterX | kLHintsExpandX, 2, 3, 0, 0));

   // camera overlay
   fCamContainer = new TGGroupFrame(fGuidesFrame, "Camera overlay");
   fGuidesFrame->AddFrame(fCamContainer, new TGLayoutHints(kLHintsTop | kLHintsCenterX | kLHintsExpandX, 2, 3, 0, 0));
   fCamOverlayOn = new TGCheckButton(fCamContainer, "Show");
   fCamContainer->AddFrame(fCamOverlayOn, new TGLayoutHints(kLHintsTop | kLHintsCenterX | kLHintsExpandX));
   TGHorizontalFrame* chf = new TGHorizontalFrame(fCamContainer);
   TGLabel* lab = new TGLabel(chf, "Mode");
   chf->AddFrame(lab, new TGLayoutHints(kLHintsLeft|kLHintsBottom, 1, 4, 1, 2));
   fCamMode = new TGComboBox(chf);
   fCamMode->AddEntry("Plane", TGLCameraOverlay::kPlaneIntersect);
   fCamMode->AddEntry("Bar", TGLCameraOverlay::kBar);
   fCamMode->AddEntry("Axis", TGLCameraOverlay::kAxis);
   fCamMode->AddEntry("Grid Front", TGLCameraOverlay::kGridFront);
   fCamMode->AddEntry("Grid Back", TGLCameraOverlay::kGridBack);
   TGListBox* lb = fCamMode->GetListBox();
   lb->Resize(lb->GetWidth(), 5*18);
   fCamMode->Resize(90, 20);
   chf->AddFrame(fCamMode, new TGLayoutHints(kLHintsTop, 1, 1, 1, 1));
   fCamContainer->AddFrame(chf);
}

//______________________________________________________________________________
void TGLViewerEditor::CreateClippingTab()
{
   // Create GUI controls - clip type (none/plane/box) and plane/box properties.

   fClipFrame = CreateEditorTabSubFrame("Clipping");

   fClipSet = new TGLClipSetSubEditor(fClipFrame);
   fClipSet->Connect("Changed()", "TGLViewerEditor", this, "ViewerRedraw()");
   fClipFrame->AddFrame(fClipSet, new TGLayoutHints(kLHintsTop | kLHintsExpandX, 2, 0, 0, 0));
}

//______________________________________________________________________________
void TGLViewerEditor::CreateStereoTab()
{
   // Create GUI controls - clip type (none/plane/box) and plane/box properties.

   fStereoFrame = CreateEditorTabSubFrame("Stereo");

   Int_t labw = 80;
   TGCompositeFrame *p = fStereoFrame;

   fStereoZeroParallax = MakeLabeledNEntry(p, "Zero parallax:", labw, 5, TGNumberFormat::kNESRealThree);
   fStereoZeroParallax->SetLimits(TGNumberFormat::kNELLimitMinMax, 0, 1);

   fStereoEyeOffsetFac = MakeLabeledNEntry(p, "Eye offset:", labw, 5, TGNumberFormat::kNESRealTwo);
   fStereoEyeOffsetFac->SetLimits(TGNumberFormat::kNELLimitMinMax, 0, 2);

   fStereoFrustumAsymFac = MakeLabeledNEntry(p, "Asymetry:", labw, 5, TGNumberFormat::kNESRealTwo);
   fStereoFrustumAsymFac->SetLimits(TGNumberFormat::kNELLimitMinMax, 0, 2);
}


//______________________________________________________________________________
void TGLViewerEditor::UpdateReferencePosState()
{
   // Enable/disable reference position (x/y/z) number edits based on
   // reference check box.

   fReferencePosX->SetState(fReferenceOn->IsDown());
   fReferencePosY->SetState(fReferenceOn->IsDown());
   fReferencePosZ->SetState(fReferenceOn->IsDown());
}

//______________________________________________________________________________
void TGLViewerEditor::SetGuides()
{
   // Configuration of guides GUI called from SetModel().

   Bool_t axesDepthTest = kFALSE;
   Bool_t referenceOn = kFALSE;
   Double_t referencePos[3] = {0.};
   fViewer->GetGuideState(fAxesType, axesDepthTest, referenceOn, referencePos);

   for (Int_t i = 1; i < 4; i++) {
      TGButton * btn = fAxesContainer->GetButton(i);
      if (fAxesType+1 == i)
         btn->SetDown(kTRUE);
      else
         btn->SetDown(kFALSE);
   }
   fAxesContainer->GetButton(4)->SetOn(axesDepthTest, kFALSE);

   fReferenceOn->SetDown(referenceOn);
   fReferencePosX->SetNumber(referencePos[0]);
   fReferencePosY->SetNumber(referencePos[1]);
   fReferencePosZ->SetNumber(referencePos[2]);
   UpdateReferencePosState();

   // overlay
   TGLCameraOverlay*  co = fViewer->GetCameraOverlay();
   TGCompositeFrame *fr = (TGCompositeFrame*)((TGFrameElement*) fCamContainer->GetList()->Last() )->fFrame;

   if (fViewer->CurrentCamera().IsOrthographic())
   {
      fCamOverlayOn->SetDown(co->GetShowOrthographic());
      fr->ShowFrame(fCamMode);


      if (! fr->IsMapped()) {
         fr->MapSubwindows();
         fr->MapWindow();
         fCamContainer->MapWindow();
         fCamContainer->MapWindow();
         fCamMode->Select(co->GetOrthographicMode(), kFALSE);
      }
   }
   else
   {
      fCamOverlayOn->SetDown(co->GetShowPerspective());

      // only mode implemented for perspective camera
      fCamMode->Select(co->GetPerspectiveMode(), kFALSE);
      fr->HideFrame(fCamMode);
      if (fr->IsMapped())
         fr->UnmapWindow();
   }
}

//______________________________________________________________________________
void TGLViewerEditor::UpdateStereo()
{
   // Update stereo related variables.

   fViewer->SetStereoZeroParallax  (fStereoZeroParallax->GetNumber());
   fViewer->SetStereoEyeOffsetFac  (fStereoEyeOffsetFac->GetNumber());
   fViewer->SetStereoFrustumAsymFac(fStereoFrustumAsymFac->GetNumber());
   ViewerRedraw(); 
}
