#include <cstring>

#include "TGedEditor.h"
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
TGLViewerEditor::TGLViewerEditor(const TGWindow *p,  Int_t width, Int_t height, UInt_t options, Pixel_t back) :
   TGedFrame(p,  width, height, options | kVerticalFrame, back),
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
  //  Constructor.

   CreateLightsTab();
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
   // Connect check buttons.
   
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


   // read lights
   UInt_t ls =  fViewer->GetLightState();
   if(ls & TGLViewer::kLightTop)
      fTopLight->SetState(kButtonDown);

   if(ls & TGLViewer::kLightRight)
      fRightLight->SetState(kButtonDown);

   if(ls & TGLViewer::kLightBottom)
      fBottomLight->SetState(kButtonDown);

   if(ls & TGLViewer::kLightLeft)
      fLeftLight->SetState(kButtonDown);

   if(ls & TGLViewer::kLightFront)
      fFrontLight->SetState(kButtonDown);
}

//______________________________________________________________________________
void TGLViewerEditor::DoButton()
{
   // Lights radio button was clicked.
   
   fViewer->ToggleLight(TGLViewer::ELight(((TGButton *) gTQSender)->WidgetId()));
}

//______________________________________________________________________________
void TGLViewerEditor::UpdateViewerGuides()
{
   // Update viewer with GUI state.
   
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
   //Creates "Lights" tab.

   fLightFrame = new TGGroupFrame(this, "Light sources:", kLHintsTop | kLHintsCenterX);

   fLightFrame->SetTitlePos(TGGroupFrame::kLeft);
   AddFrame(fLightFrame, new TGLayoutHints(kLHintsTop | kLHintsCenterX | kLHintsExpandX, 3, 3, 3, 3));//-

   TGMatrixLayout *ml = new TGMatrixLayout(fLightFrame, 0, 1, 10);
   fLightFrame->SetLayoutManager(ml);

   fTopLight = new TGCheckButton(fLightFrame, "Top", TGLViewer::kLightTop);
   fRightLight = new TGCheckButton(fLightFrame, "Right", TGLViewer::kLightRight);
   fBottomLight = new TGCheckButton(fLightFrame, "Bottom", TGLViewer::kLightBottom);
   fLeftLight = new TGCheckButton(fLightFrame, "Left", TGLViewer::kLightLeft);
   fFrontLight = new TGCheckButton(fLightFrame, "Front", TGLViewer::kLightFront);

   fLightFrame->AddFrame(fTopLight);
   fLightFrame->AddFrame(fRightLight);
   fLightFrame->AddFrame(fBottomLight);
   fLightFrame->AddFrame(fLeftLight);
   fLightFrame->AddFrame(fFrontLight);
}

//______________________________________________________________________________
void TGLViewerEditor::CreateGuidesTab()
{
   // Create "Guides" tab.
   fGuidesFrame = new TGVerticalFrame();
   AddExtraTab(new TGedSubFrame(TString("Guides"), fGuidesFrame));

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
   // Create GUI controls - clip type (none/plane/box) and plane/box properties.
   fClipFrame = new TGVerticalFrame();
   AddExtraTab(new TGedSubFrame(TString("Clipping"), fClipFrame));

   fTypeButtons = new TGButtonGroup(fClipFrame, "Clip Type");
   new TGRadioButton(fTypeButtons, "None");
   new TGRadioButton(fTypeButtons, "Plane");
   new TGRadioButton(fTypeButtons, "Box");

   fClipFrame->AddFrame(fTypeButtons, new TGLayoutHints(kLHintsTop | kLHintsCenterX | kLHintsExpandX, 3, 3, 3, 3));
   // Viewer Edit
   fEdit = new TGCheckButton(fClipFrame, "Show / Edit In Viewer", kEditId);
   fClipFrame->AddFrame(fEdit, new TGLayoutHints(kLHintsTop | kLHintsCenterX | kLHintsExpandX, 3, 3, 3, 3));

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
   fApplyButton = new TGTextButton(fClipFrame, "Apply", kApplyId);
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
   fViewer->SetClipState(fCurrentClip, data);
   fViewer->SetCurrentClip(fCurrentClip, fEdit->IsDown());
   if (fIsInPad && gGLManager)
      gGLManager->MarkForDirectCopy(fViewer->GetDev(), kTRUE);
   fViewer->RequestDraw();
}

//______________________________________________________________________________
void TGLViewerEditor::SetCurrentClip()
{
   // Set current (active) GUI clip type from 'type'.
   Bool_t edit = kFALSE;
   fViewer->GetCurrentClip(fCurrentClip, edit);
   fEdit->SetDown(edit);
   fApplyButton->SetState(kButtonDisabled);


   // Button ids run from 1
   if (TGButton *btn = fTypeButtons->GetButton(fCurrentClip+1)){
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
   // Set cintriks in "Guides" tab.
   
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
   // Hide clipping GUI.

   fClipFrame->HideFrame(fPlanePropFrame);
   fClipFrame->HideFrame(fBoxPropFrame);
}
