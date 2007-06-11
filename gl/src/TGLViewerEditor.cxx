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
#include "TGLClip.h" // should become TGLClipSetEditor
#include "TGLUtil.h"

ClassImp(TGLViewerEditor)


//A lot of raw pointers/naked new-expressions - good way to discredit C++ (or C++ programmer :) ) :(
//ROOT has system to cleanup - I'll try to use it

//______________________________________________________________________________
TGLViewerEditor::TGLViewerEditor(const TGWindow *p,  Int_t width, Int_t height, UInt_t options, Pixel_t back) :
   TGedFrame(p,  width, height, options | kVerticalFrame, back),
   fGuidesFrame(0),
   fClipFrame(0),
   fClearColor(0),
   fIgnoreSizesOnUpdate(0),
   fResetCamerasOnUpdate(0),
   fResetCameraOnDoubleClick(0),
   fUpdateScene(0),
   fCameraHome(0),
   fAxesContainer(0),
   fAxesNone(0),
   fAxesEdge(0),
   fAxesOrigin(0),
   fRefContainer(0),
   fReferenceOn(0),
   fReferencePosX(0),
   fReferencePosY(0),
   fReferencePosZ(0),
   fCamContainer(0),
   fCamMode(0),
   fCamMarkupOn(0),
   fCurrentClip(kClipNone),
   fTypeButtons(0),
   fPlanePropFrame(0),
   fPlaneProp(),
   fBoxPropFrame(0),
   fBoxProp(),
   fClipInside(0),
   fClipEdit(0),
   fClipShow(0),
   fApplyButton(0),
   fViewer(0),
   fIsInPad(kTRUE)
{
  //  Constructor.

   CreateStyleTab();
   CreateGuidesTab();
   CreateClippingTab();
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
   fResetCameraOnDoubleClick->Connect("Toggled(Bool_t)", "TGLViewerEditor", this, "DoResetCameraOnDoubleClick()");
   fUpdateScene->Connect("Pressed()", "TGLViewerEditor", this, "DoUpdateScene()");
   fCameraHome->Connect("Pressed()", "TGLViewerEditor", this, "DoCameraHome()");

   fAxesContainer->Connect("Clicked(Int_t)", "TGLViewerEditor", this, "UpdateViewerGuides()");
   fReferenceOn->Connect("Clicked()", "TGLViewerEditor", this, "UpdateViewerGuides()");
   fReferencePosX->Connect("ValueSet(Long_t)", "TGLViewerEditor", this, "UpdateViewerGuides()");
   fReferencePosY->Connect("ValueSet(Long_t)", "TGLViewerEditor", this, "UpdateViewerGuides()");
   fReferencePosZ->Connect("ValueSet(Long_t)", "TGLViewerEditor", this, "UpdateViewerGuides()");

   fCamMode->Connect("Selected(Int_t)", "TGLViewerEditor", this, "DoCameraMarkup()");
   fCamMarkupOn->Connect("Clicked()", "TGLViewerEditor", this, "DoCameraMarkup()");

   fTypeButtons->Connect("Clicked(Int_t)", "TGLViewerEditor", this, "ClipTypeChanged(Int_t)");
   fClipInside->Connect("Clicked()", "TGLViewerEditor", this, "UpdateViewerClip()");
   fClipEdit->Connect("Clicked()", "TGLViewerEditor", this, "UpdateViewerClip()");
   fClipShow->Connect("Clicked()", "TGLViewerEditor", this, "UpdateViewerClip()");

   for (Int_t i = 0; i < 4; ++i)
      fPlaneProp[i]->Connect("ValueSet(Long_t)", "TGLViewerEditor", this, "ClipValueChanged()");

   for (Int_t i = 0; i < 6; ++i)
      fBoxProp[i]->Connect("ValueSet(Long_t)", "TGLViewerEditor", this, "ClipValueChanged()");

   fApplyButton->Connect("Pressed()", "TGLViewerEditor", this, "UpdateViewerClip()");

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
   SetCurrentClip();

   if (fInit)
      ConnectSignals2Slots();

   fLightSet->SetModel(fViewer->GetLightSet());

   fClearColor->SetColor(TColor::Number2Pixel(fViewer->GetClearColor()), kFALSE);
   fIgnoreSizesOnUpdate->SetState(fViewer->GetIgnoreSizesOnUpdate() ? kButtonDown : kButtonUp);
   fResetCamerasOnUpdate->SetState(fViewer->GetResetCamerasOnUpdate() ? kButtonDown : kButtonUp);
   fResetCameraOnDoubleClick->SetState(fViewer->GetResetCameraOnDoubleClick() ? kButtonDown : kButtonUp);
}

//______________________________________________________________________________
void TGLViewerEditor::DoClearColor(Pixel_t color)
{
   // Clear-color was changed.

   fViewer->SetClearColor(Color_t(TColor::GetColor(color)));
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
void TGLViewerEditor::DoResetCameraOnDoubleClick()
{
   // ResetCameraOnDoubleClick was toggled.

   fViewer->SetResetCameraOnDoubleClick(fResetCameraOnDoubleClick->IsOn());
}

//______________________________________________________________________________
void TGLViewerEditor::DoUpdateScene()
{
   // ResetCameraOnDoubleClick was toggled.

   fViewer->UpdateScene();
}

//______________________________________________________________________________
void TGLViewerEditor::DoCameraHome()
{
   // ResetCameraOnDoubleClick was toggled.

   fViewer->ResetCurrentCamera();
   ViewerRedraw();
}

//______________________________________________________________________________
void TGLViewerEditor::DoCameraMarkup()
{
   // Update viewer with GUI state.

   TGLCameraMarkupStyle* ms = fViewer->GetCameraMarkup();
   if (ms) {
      ms->SetPosition(fCamMode->GetSelected());
      ViewerRedraw();
      ms->SetShow(fCamMarkupOn->IsDown());
   }
}

//______________________________________________________________________________
void TGLViewerEditor::UpdateViewerGuides()
{
   // Update viewer with GUI state.

   Int_t axesType = TGLUtil::kAxesNone;
   for (Int_t i = 1; i < 4; i++) {
      TGButton * button = fAxesContainer->GetButton(i);
      if (button && button->IsDown()) {
         axesType = i-1;
         break;
      }
   }

   const Double_t refPos[] = {fReferencePosX->GetNumber(), fReferencePosY->GetNumber(), fReferencePosZ->GetNumber()};
   fViewer->SetGuideState(axesType, fReferenceOn->IsDown(), refPos);
   UpdateReferencePos();
}

//______________________________________________________________________________
void TGLViewerEditor::CreateStyleTab()
{
   // Creates "Style" tab.

   // LightSet
   fLightSet = new TGLLightSetSubEditor(this);
   fLightSet->Connect("Changed", "TGLViewerEditor", this, "ViewerRedraw()");
   AddFrame(fLightSet, new TGLayoutHints(kLHintsTop | kLHintsExpandX, 2, 0, 0, 0));

   {
      TGHorizontalFrame* hf = new TGHorizontalFrame(this);
      TGLabel* lab = new TGLabel(hf, "Clear color");
      hf->AddFrame(lab, new TGLayoutHints(kLHintsLeft|kLHintsBottom, 1, 12, 1, 3));
      fClearColor = new TGColorSelect(hf, 0, -1);
      hf->AddFrame(fClearColor, new TGLayoutHints(kLHintsLeft, 1, 1, 1, 1));
      AddFrame(hf, new TGLayoutHints(kLHintsLeft, 2, 1, 1, 1));
   }

   MakeTitle("Update behaviour");
   fIgnoreSizesOnUpdate  = new TGCheckButton(this, "Ignore sizes");
   fIgnoreSizesOnUpdate->SetToolTipText("Ignore bounding-box sizes on scene update");
   AddFrame(fIgnoreSizesOnUpdate, new TGLayoutHints(kLHintsLeft, 4, 1, 1, 1));
   fResetCamerasOnUpdate = new TGCheckButton(this, "Reset on update");
   fResetCamerasOnUpdate->SetToolTipText("Reset camera on scene update");
   AddFrame(fResetCamerasOnUpdate, new TGLayoutHints(kLHintsLeft, 4, 1, 1, 1));
   fResetCameraOnDoubleClick = new TGCheckButton(this, "Reset on dbl-click");
   fResetCameraOnDoubleClick->SetToolTipText("Reset cameras on double-click");
   AddFrame(fResetCameraOnDoubleClick, new TGLayoutHints(kLHintsLeft, 4, 1, 1, 1));
   fUpdateScene = new TGTextButton(this, "Update Scene");
   AddFrame(fUpdateScene, new TGLayoutHints(kLHintsLeft|kLHintsExpandX, 4, 1, 1, 1));
   fCameraHome = new TGTextButton(this, "Camera Home");
   AddFrame(fCameraHome, new TGLayoutHints(kLHintsLeft|kLHintsExpandX, 4, 1, 1, 1));
}

//______________________________________________________________________________
void TGLViewerEditor::CreateGuidesTab()
{
   // Create "Guides" tab.
   fGuidesFrame = CreateEditorTabSubFrame("Guides");

   // axes
   fAxesContainer = new TGButtonGroup(fGuidesFrame, "Axes");
   fAxesNone = new TGRadioButton(fAxesContainer, "None");
   fAxesEdge = new TGRadioButton(fAxesContainer, "Edge");
   fAxesOrigin = new TGRadioButton(fAxesContainer, "Origin");
   fGuidesFrame->AddFrame(fAxesContainer, new TGLayoutHints(kLHintsTop | kLHintsCenterX | kLHintsExpandX, 3, 3, 3, 3));

   //Reference container
   fRefContainer = new TGGroupFrame(fGuidesFrame, "Reference Marker");
   fGuidesFrame->AddFrame(fRefContainer, new TGLayoutHints(kLHintsTop | kLHintsCenterX | kLHintsExpandX, 3, 3, 3, 3));

   //Reference options
   fReferenceOn = new TGCheckButton(fRefContainer, "Show");
   fRefContainer->AddFrame(fReferenceOn, new TGLayoutHints(kLHintsTop | kLHintsCenterX | kLHintsExpandX, 3, 3, 3, 3));

   TGLabel *label = new TGLabel(fRefContainer, "X");
   fRefContainer->AddFrame(label, new TGLayoutHints(kLHintsTop | kLHintsLeft, 0, 0, 3, 3));
   fReferencePosX = new TGNumberEntry(fRefContainer, 0.0, 8);
   fRefContainer->AddFrame(fReferencePosX, new TGLayoutHints(kLHintsTop | kLHintsCenterX | kLHintsExpandX, 3, 3, 3, 3));

   label = new TGLabel(fRefContainer, "Y");
   fRefContainer->AddFrame(label, new TGLayoutHints(kLHintsTop | kLHintsLeft, 0, 0, 3, 3));
   fReferencePosY = new TGNumberEntry(fRefContainer, 0.0, 8);
   fRefContainer->AddFrame(fReferencePosY, new TGLayoutHints(kLHintsTop | kLHintsCenterX | kLHintsExpandX, 3, 3, 3, 3));

   label = new TGLabel(fRefContainer, "Z");
   fRefContainer->AddFrame(label, new TGLayoutHints(kLHintsTop | kLHintsLeft, 0, 0, 3, 3));
   fReferencePosZ = new TGNumberEntry(fRefContainer, 0.0, 8);
   fRefContainer->AddFrame(fReferencePosZ, new TGLayoutHints(kLHintsTop | kLHintsCenterX | kLHintsExpandX, 3, 3, 3, 3));

   // camera markup
   fCamContainer = new TGGroupFrame(fGuidesFrame, "Camera Markup");
   fGuidesFrame->AddFrame(fCamContainer, new TGLayoutHints(kLHintsTop | kLHintsCenterX | kLHintsExpandX, 3, 3, 3, 3));

   fCamMarkupOn = new TGCheckButton(fCamContainer, "Show");
   fCamMarkupOn->SetToolTipText("Implemented for orthographic mode");

   fCamContainer->AddFrame(fCamMarkupOn, new TGLayoutHints(kLHintsTop | kLHintsCenterX | kLHintsExpandX, 3, 3, 3, 3));

   TGHorizontalFrame* chf = new TGHorizontalFrame(fCamContainer);
   TGLabel* lab = new TGLabel(chf, "Mode");
   chf->AddFrame(lab, new TGLayoutHints(kLHintsLeft|kLHintsBottom, 1, 4, 1, 2));
   fCamMode = new TGComboBox(chf);
   fCamMode->AddEntry("Left Up", 0);
   fCamMode->AddEntry("Left Down", 1);
   fCamMode->AddEntry("Right Up", 2);
   fCamMode->AddEntry("Right Down", 3);
   fCamMode->AddEntry("Center", 4);
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

   fTypeButtons = new TGButtonGroup(fClipFrame, "Clip Type");
   new TGRadioButton(fTypeButtons, "None");
   new TGRadioButton(fTypeButtons, "Plane");
   new TGRadioButton(fTypeButtons, "Box");

   fClipFrame->AddFrame(fTypeButtons, new TGLayoutHints(kLHintsTop | kLHintsCenterX | kLHintsExpandX, 3, 3, 3, 3));
   // Clip inside / edit
   fClipInside = new TGCheckButton(fClipFrame, "Clip away inside");
   fClipFrame->AddFrame(fClipInside, new TGLayoutHints(kLHintsTop | kLHintsCenterX | kLHintsExpandX, 3, 3, 3, 3));
   fClipEdit   = new TGCheckButton(fClipFrame, "Edit In Viewer");
   fClipFrame->AddFrame(fClipEdit, new TGLayoutHints(kLHintsTop | kLHintsCenterX | kLHintsExpandX, 3, 3, 3, 3));
   fClipShow   = new TGCheckButton(fClipFrame, "Show In Viewer");
   fClipFrame->AddFrame(fClipShow, new TGLayoutHints(kLHintsTop | kLHintsCenterX | kLHintsExpandX, 3, 3, 3, 3));

   // Plane properties
   fPlanePropFrame = new TGCompositeFrame(fClipFrame);
   //fPlanePropFrame->SetCleanup(kDeepCleanup);
   fClipFrame->AddFrame(fPlanePropFrame, new TGLayoutHints(kLHintsTop | kLHintsCenterX | kLHintsExpandX, 3, 3, 3, 3));

   static const char * const planeStr[] = { "aX + ", "bY +", "cZ + ", "d = 0" };

   for (Int_t i = 0; i < 4; ++i) {
      TGLabel *label = new TGLabel(fPlanePropFrame, planeStr[i]);
      fPlanePropFrame->AddFrame(label, new TGLayoutHints(kLHintsTop | kLHintsLeft, 3, 3, 3, 3));
      fPlaneProp[i] = new TGNumberEntry(fPlanePropFrame, 1., 8);
      fPlanePropFrame->AddFrame(fPlaneProp[i], new TGLayoutHints(kLHintsTop | kLHintsCenterX | kLHintsExpandX, 3, 3, 3, 3));
   }

   // Box properties
   fBoxPropFrame = new TGCompositeFrame(fClipFrame);
   fClipFrame->AddFrame(fBoxPropFrame, new TGLayoutHints(kLHintsTop | kLHintsCenterX | kLHintsExpandX, 3, 3, 3, 3));

   static const char * const boxStr[] = {"Center X", "Center Y", "Center Z", "Length X", "Length Y", "Length Z" };

   for (Int_t i = 0; i < 6; ++i) {
      TGLabel *label = new TGLabel(fBoxPropFrame, boxStr[i]);
      fBoxPropFrame->AddFrame(label, new TGLayoutHints(kLHintsTop | kLHintsLeft, 3, 3, 3, 3));
      fBoxProp[i] = new TGNumberEntry(fBoxPropFrame, 1., 8);
      fBoxPropFrame->AddFrame(fBoxProp[i], new TGLayoutHints(kLHintsTop | kLHintsCenterX | kLHintsExpandX, 3, 3, 3, 3));
   }

   // Apply button
   fApplyButton = new TGTextButton(fClipFrame, "Apply");
   fClipFrame->AddFrame(fApplyButton, new TGLayoutHints(kLHintsTop | kLHintsCenterX | kLHintsExpandX, 3, 3, 3, 3));

}

//______________________________________________________________________________
void TGLViewerEditor::UpdateReferencePos()
{
   // Enable/disable reference position (x/y/z) number edits based on
   // reference check box.

   fReferencePosX->SetState(fReferenceOn->IsDown());
   fReferencePosY->SetState(fReferenceOn->IsDown());
   fReferencePosZ->SetState(fReferenceOn->IsDown());
}

//______________________________________________________________________________
void TGLViewerEditor::ClipValueChanged()
{
   // One of number edtries was changed.

   fApplyButton->SetState(kButtonUp);
}

//______________________________________________________________________________
void TGLViewerEditor::ClipTypeChanged(Int_t id)
{
   // Clip type radio button changed - update viewer.

   TGLClipSet * clipset = fViewer->GetClipSet();
   if (id == 1) {
      fCurrentClip = kClipNone;
      clipset->SetClipType(fCurrentClip);
      SetCurrentClip();
      fClipInside->SetState(kButtonDisabled);
      fClipEdit->SetState(kButtonDisabled);
      fClipShow->SetState(kButtonDisabled);
   } else {
      fCurrentClip = id == 2 ? kClipPlane : kClipBox;
      clipset->SetClipType(fCurrentClip);
      SetCurrentClip();
   }

   // Internal GUI change - need to update the viewer
   ViewerRedraw();

   fGedEditor->Layout();
}

//______________________________________________________________________________
void TGLViewerEditor::UpdateViewerClip()
{
   // Change clipping volume.

   Double_t data[6] = {0.};
   // Fetch GUI state for clip if 'type' into 'data' vector
   if (fCurrentClip == kClipPlane)
      for (Int_t i = 0; i < 4; ++i)
         data[i] = fPlaneProp[i]->GetNumber();
   else if (fCurrentClip == kClipBox)
      for (Int_t i = 0; i < 6; ++i)
         data[i] = fBoxProp[i]->GetNumber();

   fApplyButton->SetState(kButtonDisabled);
   TGLClipSet * clipset = fViewer->GetClipSet();
   clipset->SetClipState(fCurrentClip, data);
   clipset->SetShowManip(fClipEdit->IsDown());
   clipset->SetShowClip (fClipShow->IsDown());
   if (fCurrentClip != kClipNone)
      clipset->GetCurrentClip()->SetMode(fClipInside->IsDown() ? TGLClip::kInside : TGLClip::kOutside);
   ViewerRedraw();
}

//______________________________________________________________________________
void TGLViewerEditor::SetCurrentClip()
{
   // Set current (active) GUI clip type from viewer' state.

   Double_t     clip[6] = {0.};
   TGLClipSet * clipset = fViewer->GetClipSet();

   clipset->GetClipState(fCurrentClip, clip);

   fClipEdit->SetDown(clipset->GetShowManip());
   fClipShow->SetDown(clipset->GetShowClip());
   if (fCurrentClip != kClipNone)
      fClipInside->SetDown(clipset->GetCurrentClip()->GetMode() == TGLClip::kInside);

   fApplyButton->SetState(kButtonDisabled);

   // Button ids run from 1
   if (TGButton *btn = fTypeButtons->GetButton(fCurrentClip+1)) {
      btn->SetDown();
   }
   switch(fCurrentClip) {
   case(kClipNone):
      fTypeButtons->SetButton(1);
      fClipFrame->HideFrame(fPlanePropFrame);
      fClipFrame->HideFrame(fBoxPropFrame);
      return;
   case(kClipPlane):
      fTypeButtons->SetButton(2);
      fClipFrame->ShowFrame(fPlanePropFrame);
      fClipFrame->HideFrame(fBoxPropFrame);
      break;
   case(kClipBox):
      fTypeButtons->SetButton(3);
      fClipFrame->HideFrame(fPlanePropFrame);
      fClipFrame->ShowFrame(fBoxPropFrame);
      break;
   default:;
   }

   if (fCurrentClip == kClipPlane)
      for (Int_t i = 0; i < 4; ++i)
         fPlaneProp[i]->SetNumber(clip[i]);
   else if (fCurrentClip == kClipBox)
      for (Int_t i = 0; i < 6; ++i)
         fBoxProp[i]->SetNumber(clip[i]);

   ViewerRedraw();
}

//______________________________________________________________________________
void TGLViewerEditor::SetGuides()
{
   // Set cintriks in "Guides" tab.

   Int_t axesType = TGLUtil::kAxesNone;
   Bool_t referenceOn = kFALSE;
   Double_t referencePos[3] = {0.};
   fViewer->GetGuideState(axesType, referenceOn, referencePos);

   // Button ids run from 1
   if (TGButton *btn = fAxesContainer->GetButton(axesType+1))
      btn->SetDown();

   fReferenceOn->SetDown(referenceOn);
   fReferencePosX->SetNumber(referencePos[0]);
   fReferencePosY->SetNumber(referencePos[1]);
   fReferencePosZ->SetNumber(referencePos[2]);
   UpdateReferencePos();

   if (fViewer->CurrentCamera().IsA()->InheritsFrom("TGLOrthoCamera")) {
      fGuidesFrame->ShowFrame(fCamContainer);
      fCamMarkupOn->SetDown(fViewer->GetCameraMarkup()->Show());
      fCamMode->Select(fViewer->GetCameraMarkup()->Position(), kFALSE);
   } else {
      fGuidesFrame->HideFrame(fCamContainer);
   }
}

//______________________________________________________________________________
void TGLViewerEditor::HideClippingGUI()
{
   // Hide clipping GUI.

   fClipFrame->HideFrame(fPlanePropFrame);
   fClipFrame->HideFrame(fBoxPropFrame);
}
