#include <cstring>

#include "TGNumberEntry.h"
#include "TGButtonGroup.h"
#include "TVirtualGL.h"
#include "TG3DLine.h"
#include "TGButton.h"
#include "TString.h"
#include "TGLabel.h"
#include "TClass.h"
#include "TGTab.h"

#include "TGLViewerEditor.h"
#include "TGLViewer.h"
#include "TGLUtil.h"

ClassImp(TGLViewerEditor)

//A lot of raw pointers/naked new-expressions - good way to discredit C++ (or C++ programmer :) ) :(
//ROOT has system to cleanup - I'll try to use it

//______________________________________________________________________________
TGLViewerEditor::TGLViewerEditor(const TGWindow *p, Int_t id, Int_t width, Int_t height, UInt_t options, Pixel_t back)
                  : TGedFrame(p, id, width, height, options | kVerticalFrame, back),
                    fGuidesTabEl(0),
                    fClipTabEl(0),
                    fGuidesFrame(0),
                    fClipFrame(0),
                    fLightFrame(0),
                    fTopLight(0),
                    fRightLight(0),
                    fBottomLight(0),
                    fLeftLight(0),
                    fFrontLight(0),
                    fAxesContainer(0),
                    fAxesNone(0),
                    fAxesEdge(0),
                    fAxesOrigin(0),
                    fRefContainer(0),
                    fReferenceOn(0),
                    fReferencePosX(0),
                    fReferencePosY(0),
                    fReferencePosZ(0),
                    fCurrentClip(kClipNone),
                    fTypeButtons(0),
                    fPlanePropFrame(0),
                    fPlaneProp(),
                    fBoxPropFrame(0),
                    fBoxProp(),
                    fEdit(0),
                    fApplyButton(0),
                    fViewer(0),
                    fIsInPad(kTRUE)
{
   //Create tabs
   CreateLightsTab();
   CreateGuidesTab();
   CreateClippingTab();

   fTab->Layout();
   fTab->MapSubwindows();
   //Register editor
   TGedElement *ged = new TGedElement;
   ged->fGedFrame = this;
   ged->fCanvas = 0;
   TGLViewer::Class()->GetEditorList()->Add(ged);
}

//______________________________________________________________________________
TGLViewerEditor::TGLViewerEditor(const TGWindow *p)
                  : TGedFrame(p, 0, 140, 30, kChildFrame | kVerticalFrame, GetDefaultFrameBackground()),
                    fGuidesTabEl(0),
                    fClipTabEl(0),
                    fGuidesFrame(0),
                    fClipFrame(0),
                    fLightFrame(0),
                    fTopLight(0),
                    fRightLight(0),
                    fBottomLight(0),
                    fLeftLight(0),
                    fFrontLight(0),
                    fAxesContainer(0),
                    fAxesNone(0),
                    fAxesEdge(0),
                    fAxesOrigin(0),
                    fRefContainer(0),
                    fReferenceOn(0),
                    fReferencePosX(0),
                    fReferencePosY(0),
                    fReferencePosZ(0),
                    fCurrentClip(kClipNone),
                    fTypeButtons(0),
                    fPlanePropFrame(0),
                    fPlaneProp(),
                    fBoxPropFrame(0),
                    fBoxProp(),
                    fEdit(0),
                    fApplyButton(0),
                    fViewer(0),
                    fIsInPad(kFALSE)
{
   //Create tabs
   CreateLightsTab();
   CreateGuidesTab();
   CreateClippingTab();

   fTab->Layout();
   fTab->MapSubwindows();
}

//______________________________________________________________________________
TGLViewerEditor::~TGLViewerEditor()
{
   //Try to cleanup
   SetCleanup(kDeepCleanup);
   Cleanup();

   fGuidesFrame->SetCleanup(kDeepCleanup);
   fGuidesFrame->Cleanup();

   fClipFrame->SetCleanup(kDeepCleanup);
   fClipFrame->Cleanup();
}

//______________________________________________________________________________
void TGLViewerEditor::ConnectSignals2Slots()
{
   //Connect check buttons
   //Do I really need this function, or I can connect directly in CreateXXXTab functions ???
   fTopLight->Connect("Clicked()", "TGLViewerEditor", this, "DoButton()");
   fRightLight->Connect("Clicked()", "TGLViewerEditor", this, "DoButton()");
   fBottomLight->Connect("Clicked()", "TGLViewerEditor", this, "DoButton()");
   fLeftLight->Connect("Clicked()", "TGLViewerEditor", this, "DoButton()");
   fFrontLight->Connect("Clicked()", "TGLViewerEditor", this, "DoButton()");

   fAxesContainer->Connect("Pressed(Int_t)", "TGLViewerEditor", this, "UpdateViewerGuides()");
   fReferenceOn->Connect("Clicked()", "TGLViewerEditor", this, "UpdateViewerGuides()");
   fReferencePosX->Connect("ValueSet(Long_t)", "TGLViewerEditor", this, "UpdateViewerGuides()");
   fReferencePosY->Connect("ValueSet(Long_t)", "TGLViewerEditor", this, "UpdateViewerGuides()");
   fReferencePosZ->Connect("ValueSet(Long_t)", "TGLViewerEditor", this, "UpdateViewerGuides()");

   fTypeButtons->Connect("Pressed(Int_t)", "TGLViewerEditor", this, "ClipTypeChanged(Int_t)");
   fEdit->Connect("Clicked()", "TGLViewerEditor", this, "UpdateViewerClip()");

   for (Int_t i = 0; i < 4; ++i)
      fPlaneProp[i]->Connect("ValueSet(Long_t)", "TGLViewerEditor", this, "ClipValueChanged()");

   for (Int_t i = 0; i < 6; ++i)
      fBoxProp[i]->Connect("ValueSet(Long_t)", "TGLViewerEditor", this, "ClipValueChanged()");

   fApplyButton->Connect("Pressed()", "TGLViewerEditor", this, "UpdateViewerClip()");

   fInit = kFALSE;
}

//______________________________________________________________________________
void TGLViewerEditor::SetModel(TVirtualPad *pad, TObject *obj, Int_t)
{
   //Sets model or disables/hides viewer
   fViewer = 0;
   fModel = 0;
   fPad = 0;

   if (!obj || !obj->InheritsFrom(TGLViewer::Class())) {
      SetActive(kFALSE);
      fGuidesTabEl->UnmapWindow();
      fGuidesFrame->UnmapWindow();
      fClipTabEl->UnmapWindow();
      fClipFrame->UnmapWindow();
      fTab->SetTab(0);
      return;
   } else {
      fGuidesTabEl->MapWindow();
      fGuidesFrame->MapWindow();
      fClipTabEl->MapWindow();
      fClipFrame->MapWindow();
   }

   fViewer = static_cast<TGLViewer *>(obj);
   if (fIsInPad)
      fViewer->SetPadEditor(this);
   fModel = obj;
   fPad = pad;
   //Set guides controls' values
   SetGuides();
   //Set clipping control values
   SetCurrentClip();

   if (fInit)
      ConnectSignals2Slots();

   SetActive();
}

//______________________________________________________________________________
void TGLViewerEditor::DoButton()
{
   //Lights radio button was clicked
   fViewer->ToggleLight(TGLViewer::ELight(((TGButton *) gTQSender)->WidgetId()));
}

//______________________________________________________________________________
void TGLViewerEditor::UpdateViewerGuides()
{
   // Update viewer with GUI state
   TGLViewer::EAxesType axesType = TGLViewer::kAxesNone;
   for (Int_t i = 1; i < 4; i++) {
      TGButton * button = fAxesContainer->GetButton(i);
      if (button && button->IsDown()) {
         axesType = TGLViewer::EAxesType(i-1);
         break;
      }
   }

   const Double_t refPos[] = {fReferencePosX->GetNumber(), fReferencePosY->GetNumber(), fReferencePosZ->GetNumber()};
   fViewer->SetGuideState(axesType, fReferenceOn->IsDown(), refPos);
   UpdateReferencePos();
}

//______________________________________________________________________________
void TGLViewerEditor::CreateLightsTab()
{
   //Creates "Lights" tab
   fLightFrame = new TGGroupFrame(this, "Light sources:", kLHintsTop | kLHintsCenterX);
   fLightFrame->SetCleanup(kDeepCleanup);
   fLightFrame->SetTitlePos(TGGroupFrame::kLeft);
   AddFrame(fLightFrame, new TGLayoutHints(kLHintsTop | kLHintsCenterX | kLHintsExpandX, 3, 3, 3, 3));//-

   TGMatrixLayout *ml = new TGMatrixLayout(fLightFrame, 0, 1, 10);
   fLightFrame->SetLayoutManager(ml);

   fTopLight = new TGCheckButton(fLightFrame, "Top", TGLViewer::kLightTop);
   fTopLight->SetState(kButtonDown);
   fRightLight = new TGCheckButton(fLightFrame, "Right", TGLViewer::kLightRight);
   fRightLight->SetState(kButtonDown);
   fBottomLight = new TGCheckButton(fLightFrame, "Bottom", TGLViewer::kLightBottom);
   fBottomLight->SetState(kButtonDown);
   fLeftLight = new TGCheckButton(fLightFrame, "Left", TGLViewer::kLightLeft);
   fLeftLight->SetState(kButtonDown);
   fFrontLight = new TGCheckButton(fLightFrame, "Front", TGLViewer::kLightFront);
   fFrontLight->SetState(kButtonDown);

   fLightFrame->AddFrame(fTopLight);
   fLightFrame->AddFrame(fRightLight);
   fLightFrame->AddFrame(fBottomLight);
   fLightFrame->AddFrame(fLeftLight);
   fLightFrame->AddFrame(fFrontLight);
}

//______________________________________________________________________________
void TGLViewerEditor::CreateGuidesTab()
{
   //Create "Guides" tab
   fGuidesFrame = fTab->AddTab("Guides");
   fGuidesTabEl = fTab->GetTabTab("Guides");

   TGCompositeFrame *nameBin = new TGCompositeFrame(fGuidesFrame, 145, 10, kHorizontalFrame | kFixedWidth | kOwnBackground);
   nameBin->SetCleanup(kDeepCleanup);
   nameBin->AddFrame(new TGLabel(nameBin,"Name"), new TGLayoutHints(kLHintsLeft, 1, 1, 5, 0));
   nameBin->AddFrame(new TGHorizontal3DLine(nameBin), new TGLayoutHints(kLHintsExpandX, 5, 5, 12, 7));

   fGuidesFrame->AddFrame(nameBin, new TGLayoutHints(kLHintsTop, 1, 1, 0, 0));
   TGLabel *nameLabel = new TGLabel(fGuidesFrame, "TGLViewer::TGLViewer");
   Pixel_t color;
   gClient->GetColorByName("#ff0000", color);
   nameLabel->SetTextColor(color, kFALSE);
   fGuidesFrame->AddFrame(nameLabel, new TGLayoutHints(kLHintsLeft, 1, 1, 0, 0));

   fAxesContainer = new TGButtonGroup(fGuidesFrame, "Axes");
   fAxesContainer->SetCleanup(kDeepCleanup);

   fAxesNone = new TGRadioButton(fAxesContainer, "None");
   fAxesEdge = new TGRadioButton(fAxesContainer, "Edge");
   fAxesOrigin = new TGRadioButton(fAxesContainer, "Origin");

   fGuidesFrame->AddFrame(fAxesContainer, new TGLayoutHints(kLHintsTop | kLHintsCenterX | kLHintsExpandX, 3, 3, 3, 3));
   //Reference container
   fRefContainer = new TGGroupFrame(fGuidesFrame, "Reference Marker");
   fRefContainer->SetCleanup(kDeepCleanup);
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
}

namespace
{
   enum EClippingControlIds {
      kEditId,
      kApplyId
   };
}

//______________________________________________________________________________
void TGLViewerEditor::CreateClippingTab()
{
   // Create GUI controls - clip type (none/plane/box) and plane/box properties
   fClipFrame = fTab->AddTab("Clipping");
   fClipTabEl = fTab->GetTabTab("Clipping");
   //
   TGCompositeFrame *nameBin = new TGCompositeFrame(fClipFrame, 145, 10, kHorizontalFrame | kFixedWidth | kOwnBackground);
   nameBin->SetCleanup(kDeepCleanup);
   nameBin->AddFrame(new TGLabel(nameBin,"Name"), new TGLayoutHints(kLHintsLeft, 1, 1, 5, 0));
   nameBin->AddFrame(new TGHorizontal3DLine(nameBin), new TGLayoutHints(kLHintsExpandX, 5, 5, 12, 7));

   fClipFrame->AddFrame(nameBin, new TGLayoutHints(kLHintsTop, 1, 1, 0, 0));
   TGLabel *nameLabel = new TGLabel(fClipFrame, "TGLViewer::TGLViewer");
   Pixel_t color;
   gClient->GetColorByName("#ff0000", color);
   nameLabel->SetTextColor(color, kFALSE);
   fClipFrame->AddFrame(nameLabel, new TGLayoutHints(kLHintsLeft, 1, 1, 0, 0));

   fTypeButtons = new TGButtonGroup(fClipFrame, "Clip Type");
   fTypeButtons->SetCleanup(kDeepCleanup);
   new TGRadioButton(fTypeButtons, "None");
   new TGRadioButton(fTypeButtons, "Plane");
   new TGRadioButton(fTypeButtons, "Box");

   fClipFrame->AddFrame(fTypeButtons, new TGLayoutHints(kLHintsTop | kLHintsCenterX | kLHintsExpandX, 3, 3, 3, 3));
   // Viewer Edit
   fEdit = new TGCheckButton(fClipFrame, "Show / Edit In Viewer", kEditId);
   fClipFrame->AddFrame(fEdit, new TGLayoutHints(kLHintsTop | kLHintsCenterX | kLHintsExpandX, 3, 3, 3, 3));

   // Plane properties
   fPlanePropFrame = new TGCompositeFrame(fClipFrame);
   fPlanePropFrame->SetCleanup(kDeepCleanup);
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
   fBoxPropFrame->SetCleanup(kDeepCleanup);
   fClipFrame->AddFrame(fBoxPropFrame, new TGLayoutHints(kLHintsTop | kLHintsCenterX | kLHintsExpandX, 3, 3, 3, 3));

   static const char * const boxStr[] = {"Center X", "Center Y", "Center Z", "Length X", "Length Y", "Length Z" };

   for (Int_t i = 0; i < 6; ++i) {
      TGLabel *label = new TGLabel(fBoxPropFrame, boxStr[i]);
      fBoxPropFrame->AddFrame(label, new TGLayoutHints(kLHintsTop | kLHintsLeft, 3, 3, 3, 3));
      fBoxProp[i] = new TGNumberEntry(fBoxPropFrame, 1., 8);
      fBoxPropFrame->AddFrame(fBoxProp[i], new TGLayoutHints(kLHintsTop | kLHintsCenterX | kLHintsExpandX, 3, 3, 3, 3));
   }

   // Apply button
   fApplyButton = new TGTextButton(fClipFrame, "Apply", kApplyId);
   fClipFrame->AddFrame(fApplyButton, new TGLayoutHints(kLHintsTop | kLHintsCenterX | kLHintsExpandX, 3, 3, 3, 3));
}

//______________________________________________________________________________
void TGLViewerEditor::UpdateReferencePos()
{
   // Enable/disable reference position (x/y/z) number edits based on
   // reference check box
   fReferencePosX->SetState(fReferenceOn->IsDown());
   fReferencePosY->SetState(fReferenceOn->IsDown());
   fReferencePosZ->SetState(fReferenceOn->IsDown());
}

//______________________________________________________________________________
void TGLViewerEditor::ClipValueChanged()
{
   //One of number edtries was changed
   fApplyButton->SetState(kButtonUp);
}

//______________________________________________________________________________
void TGLViewerEditor::ClipTypeChanged(Int_t id)
{
   // Clip type radio button changed - update viewer
   if (id == 1) {
      fCurrentClip = kClipNone;
      fViewer->SetCurrentClip(kClipNone, kFALSE);
      SetCurrentClip();
      fEdit->SetState(kButtonDisabled);
   } else {
      fEdit->SetState(kButtonUp);
      fCurrentClip = id == 2 ? kClipPlane : kClipBox;
      fViewer->SetCurrentClip(fCurrentClip, fEdit->IsDown());
      SetCurrentClip();
   }

   // Internal GUI change - need to update the viewer
   if (gGLManager && fIsInPad)
      gGLManager->MarkForDirectCopy(fViewer->GetDev(), kTRUE);
   fViewer->RequestDraw();
}

//______________________________________________________________________________
void TGLViewerEditor::UpdateViewerClip()
{
   //Change clipping volume
   Double_t data[6] = {0.};
   // Fetch GUI state for clip if 'type' into 'data' vector
   if (fCurrentClip == kClipPlane)
      for (Int_t i = 0; i < 4; ++i)
         data[i] = fPlaneProp[i]->GetNumber();
   else if (fCurrentClip == kClipBox)
      for (Int_t i = 0; i < 6; ++i)
         data[i] = fBoxProp[i]->GetNumber();

   fApplyButton->SetState(kButtonDisabled);
   fViewer->SetClipState(fCurrentClip, data);
   fViewer->SetCurrentClip(fCurrentClip, fEdit->IsDown());
   if (fIsInPad && gGLManager)
      gGLManager->MarkForDirectCopy(fViewer->GetDev(), kTRUE);
   fViewer->RequestDraw();
}

//______________________________________________________________________________
void TGLViewerEditor::SetCurrentClip()
{
   // Set current (active) GUI clip type from 'type'
   Bool_t edit = kFALSE;
   fViewer->GetCurrentClip(fCurrentClip, edit);
   fEdit->SetDown(edit);
   fApplyButton->SetState(kButtonDisabled);

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

   Double_t clip[6] = {0.};
   fViewer->GetClipState(fCurrentClip, clip);

   if (fCurrentClip == kClipPlane)
      for (Int_t i = 0; i < 4; ++i)
         fPlaneProp[i]->SetNumber(clip[i]);
   else if (fCurrentClip == kClipBox)
      for (Int_t i = 0; i < 6; ++i)
         fBoxProp[i]->SetNumber(clip[i]);

   if (fIsInPad && gGLManager)
      gGLManager->MarkForDirectCopy(fViewer->GetDev(), kTRUE);
   fViewer->RequestDraw();
}

//______________________________________________________________________________
void TGLViewerEditor::SetGuides()
{
   //Set cintriks in "Guides" tab
   TGLViewer::EAxesType axesType = TGLViewer::kAxesNone;
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
}

//______________________________________________________________________________
void TGLViewerEditor::HideClippingGUI()
{
   fClipFrame->HideFrame(fPlanePropFrame);
   fClipFrame->HideFrame(fBoxPropFrame);
}
