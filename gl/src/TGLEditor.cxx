// @(#)root/gl:$Name:  $:$Id: TGLEditor.cxx,v 1.27 2006/03/13 09:33:50 brun Exp $
// Author:  Timur Pocheptsov  03/08/2004

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/
#include "TVirtualGL.h"
#include "TVirtualX.h"
#include "TGCanvas.h"
#include "TGLayout.h"
#include "TGButton.h"
#include "TGButtonGroup.h"
#include "TGSlider.h"
#include "TGLabel.h"
#include "TGNumberEntry.h"
#include "TGLSAViewer.h"

#include "TGLEditor.h"

class TGLMatView : public TGCompositeFrame {
private:
   TGLColorEditor *fOwner;
public:
   TGLMatView(const TGWindow *parent, Window_t wid, TGLColorEditor *owner);
   Bool_t HandleConfigureNotify(Event_t *event);
   Bool_t HandleExpose(Event_t *event);

private:
   TGLMatView(const TGLMatView &);
   TGLMatView & operator = (const TGLMatView &);
};

//______________________________________________________________________________
TGLMatView::TGLMatView(const TGWindow *parent, Window_t wid, TGLColorEditor *owner)
               :TGCompositeFrame(gClient, wid, parent), fOwner(owner)
{
   //
   AddInput(kExposureMask | kStructureNotifyMask);
}

//______________________________________________________________________________
Bool_t TGLMatView::HandleConfigureNotify(Event_t *event)
{
   //
   return fOwner->HandleContainerNotify(event);
}

//______________________________________________________________________________
Bool_t TGLMatView::HandleExpose(Event_t *event)
{
   //
   return fOwner->HandleContainerExpose(event);
}

namespace {

   enum EGLEditorIdent {
      kCPa = kTBa1 + 1,
      kCPd, kCPs, kCPe,
      kHSr, kHSg, kHSb,
      kHSa, kHSs, kHSe,
      kNExc, kNEyc, kNEzc,
      kNExs, kNEys, kNEzs,
      kNExp, kNEyp, kNEzp,
      kNEat
   };

}

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGLColorEditor                                                       //
//                                                                      //
// GL Viewer shape color editor GUI component                           //
//////////////////////////////////////////////////////////////////////////

ClassImp(TGLColorEditor)

//______________________________________________________________________________
TGLColorEditor::TGLColorEditor(const TGWindow *parent, TGLSAViewer *v)
               :TGCompositeFrame(parent, 100, 100, kVerticalFrame),// | kRaisedFrame),
                fViewer(v), fRedSlider(0), fGreenSlider(0), fBlueSlider(0),
                fAlphaSlider(0), fApplyButton(0), fIsActive(kFALSE),
                fIsLight(kFALSE), fRGBA()
{
   // Construct color editor GUI component, parented by window 'parent',
   // bound to viewer 'v'
   for (Int_t i = 0; i < 12; ++i) fRGBA[i] = 1.;

   fRGBA[12] = 0.f, fRGBA[13] = 0.f, fRGBA[14] = 0.f;
   fRGBA[15] = 1.f, fRGBA[16] = 60.f;

   CreateMaterialView();
   CreateRadioButtons();
   CreateSliders();

   //apply button creation
   fApplyButton = new TGTextButton(this, "Apply", kTBa);
   AddFrame(fApplyButton, new TGLayoutHints(kLHintsTop | kLHintsLeft | kLHintsExpandX, 2, 2, 5, 0));
   fApplyButton->SetState(kButtonDisabled);
   fApplyButton->Connect("Pressed()", "TGLColorEditor", this, "DoButton()");
   //apply to family button creation
   fApplyFamily = new TGTextButton(this, "Apply to family", kTBaf);
   AddFrame(fApplyFamily, new TGLayoutHints(kLHintsTop | kLHintsLeft | kLHintsExpandX, 2, 2, 5, 0));
   fApplyFamily->SetState(kButtonDisabled);
   fApplyFamily->Connect("Pressed()", "TGLColorEditor", this, "DoButton()");

   //Init GL
   MakeCurrent();

   gVirtualGL->NewPRGL();
   gVirtualGL->FrustumGL(-0.5, 0.5, -0.5, 0.5, 1., 10.);
   gVirtualGL->EnableGL(kLIGHTING);
   gVirtualGL->EnableGL(kLIGHT0);
   gVirtualGL->EnableGL(kDEPTH_TEST);
   gVirtualGL->EnableGL(kCULL_FACE);
   gVirtualGL->CullFaceGL(kBACK);

   DrawSphere();
}

//______________________________________________________________________________
TGLColorEditor::~TGLColorEditor()
{
   // Destroy color editor GUI component
   delete fMatView;
   gVirtualGL->DeleteContext(fCtx);
}

//______________________________________________________________________________
void TGLColorEditor::SetRGBA(const Float_t *rgba)
{
   // Set color sliders from 17 component 'rgba'
   fApplyButton->SetState(kButtonDisabled);
   fApplyFamily->SetState(kButtonDisabled);

   fIsActive = kTRUE;

   for (Int_t i = 0; i < 17; ++i) fRGBA[i] = rgba[i];

   if (rgba[16] < 0.f) {//this conditional part is obsolete now, we cannot edit ligts more
      if (fLMode == kEmission) {
         fLMode = kDiffuse;
         fLightTypes[kDiffuse]->SetState(kButtonDown);
         fLightTypes[kEmission]->SetState(kButtonUp);
      }
      fLightTypes[kEmission]->SetState(kButtonDisabled);
      fIsLight = kTRUE;
   } else {
      fIsLight = kFALSE;
      fLightTypes[kEmission]->SetState(kButtonUp);
      fAlphaSlider->SetPosition(Int_t(fRGBA[3] * 100));
      fShineSlider->SetPosition(Int_t(fRGBA[16]));
   }

   fRedSlider->SetPosition(Int_t(fRGBA[fLMode * 4] * 100));
   fGreenSlider->SetPosition(Int_t(fRGBA[fLMode * 4 + 1] * 100));
   fBlueSlider->SetPosition(Int_t(fRGBA[fLMode * 4 + 2] * 100));

   DrawSphere();
}

//______________________________________________________________________________
void TGLColorEditor::DoSlider(Int_t val)
{
   // Process slider movement
   TGSlider *frm = (TGSlider *)gTQSender;

   if (frm) {
      Int_t wid = frm->WidgetId();

      switch (wid) {
      case kHSr:
         fRGBA[fLMode * 4] = val / 100.f;
         break;
      case kHSg:
         fRGBA[fLMode * 4 + 1] = val / 100.f;
         break;
      case kHSb:
         fRGBA[fLMode * 4 + 2] = val / 100.f;
         break;
      case kHSa:
         if (!fIsLight) fRGBA[fLMode * 4 + 3] = val / 100.f;
         break;
      case kHSs:
         if (!fIsLight) fRGBA[16] = val;
         break;
      }

      if (!fIsLight || (wid != kHSa && wid != kHSs)) {
         if (fIsActive) {
            fApplyButton->SetState(kButtonUp);
            if (!fIsLight) fApplyFamily->SetState(kButtonUp);
         }
         DrawSphere();
      }
   }
}

//______________________________________________________________________________
void TGLColorEditor::DoButton()
{
   // Process button action
   TGButton *btn = (TGButton *) gTQSender;
   Int_t id = btn->WidgetId();

   switch (id) {
   case kCPd:
      fLightTypes[fLMode]->SetState(kButtonUp);
      fLMode = kDiffuse;
      SetSlidersPos();
      break;
   case kCPa:
      fLightTypes[fLMode]->SetState(kButtonUp);
      fLMode = kAmbient;
      SetSlidersPos();
      break;
   case kCPs:
      fLightTypes[fLMode]->SetState(kButtonUp);
      fLMode = kSpecular;
      SetSlidersPos();
      break;
   case kCPe:
      fLightTypes[fLMode]->SetState(kButtonUp);
      fLMode = kEmission;
      SetSlidersPos();
      break;
   case kTBa:
   case kTBaf:
      fApplyButton->SetState(kButtonDisabled);
      fApplyFamily->SetState(kButtonDisabled);
      fViewer->ProcessGUIEvent(id);
      break;
   }
   DrawSphere();
}

//______________________________________________________________________________
void TGLColorEditor::Disable()
{
   // Disable 'Apply' button and internal flags
   fApplyButton->SetState(kButtonDisabled);
   fApplyButton->SetState(kButtonDisabled);
   fIsActive = kFALSE;
   fIsLight = kFALSE;
   DrawSphere();
}

//______________________________________________________________________________
void TGLColorEditor::CreateMaterialView()
{
   //Small gl-window with sphere
   TGCanvas *viewCanvas = new TGCanvas(this, 120, 120, kSunkenFrame | kDoubleBorder);
   Window_t wid = viewCanvas->GetViewPort()->GetId();
   fGLWin = gVirtualGL->CreateGLWindow(wid);

   fMatView = new TGLMatView(viewCanvas->GetViewPort(), fGLWin, this);
   fCtx = gVirtualGL->CreateContext(fGLWin);
   viewCanvas->SetContainer(fMatView);
   AddFrame(viewCanvas, new TGLayoutHints(kLHintsTop | kLHintsCenterX, 2, 0, 2, 2));
}

//______________________________________________________________________________
void TGLColorEditor::CreateRadioButtons()
{
   // Create Diffuse/Ambient/Specular/Emissive radio buttons and sub-frames
   TGGroupFrame *partFrame = new TGGroupFrame(this, "Color components:", kLHintsTop | kLHintsCenterX);
   partFrame->SetTitlePos(TGGroupFrame::kLeft);
   AddFrame(partFrame, new TGLayoutHints(kLHintsTop | kLHintsCenterX, 2, 0, 2, 2));
   TGMatrixLayout *ml = new TGMatrixLayout(partFrame, 0, 1, 10);
   partFrame->SetLayoutManager(ml);

   // partFrame will delete the layout manager ml for us so don't add to fTrash
   fLightTypes[kDiffuse] = new TGRadioButton(partFrame, "Diffuse", kCPd);
   fLightTypes[kDiffuse]->Connect("Pressed()", "TGLColorEditor", this, "DoButton()");
   fLightTypes[kDiffuse]->SetToolTipText("Diffuse component of color");
   partFrame->AddFrame(fLightTypes[kDiffuse]);

   fLightTypes[kAmbient] = new TGRadioButton(partFrame, "Ambient", kCPa);
   fLightTypes[kAmbient]->Connect("Pressed()", "TGLColorEditor", this, "DoButton()");
   fLightTypes[kAmbient]->SetToolTipText("Ambient component of color");
   partFrame->AddFrame(fLightTypes[kAmbient]);

   fLightTypes[kSpecular] = new TGRadioButton(partFrame, "Specular", kCPs);
   fLightTypes[kSpecular]->Connect("Pressed()", "TGLColorEditor", this, "DoButton()");
   fLightTypes[kSpecular]->SetToolTipText("Specular component of color");
   partFrame->AddFrame(fLightTypes[kSpecular]);

   fLightTypes[kEmission] = new TGRadioButton(partFrame, "Emissive", kCPe);
   fLightTypes[kEmission]->Connect("Pressed()", "TGLColorEditor", this, "DoButton()");
   fLightTypes[kEmission]->SetToolTipText("Emissive component of color");
   partFrame->AddFrame(fLightTypes[kEmission]);

   fLMode = kDiffuse;
   fLightTypes[fLMode]->SetState(kButtonDown);
}

//______________________________________________________________________________
void TGLColorEditor::CreateSliders()
{
   // Create Red/Green/BlueAlpha/Shine sliders
   fRedSlider = new TGHSlider(this, 100, kSlider1 | kScaleBoth, kHSr);
   fRedSlider->Connect("PositionChanged(Int_t)", "TGLColorEditor", this, "DoSlider(Int_t)");
   fRedSlider->SetRange(0, 100);
   fRedSlider->SetPosition(Int_t(fRGBA[0] * 100));

   fGreenSlider = new TGHSlider(this, 100, kSlider1 | kScaleBoth, kHSg);
   fGreenSlider->Connect("PositionChanged(Int_t)", "TGLColorEditor", this, "DoSlider(Int_t)");
   fGreenSlider->SetRange(0, 100);
   fGreenSlider->SetPosition(Int_t(fRGBA[1] * 100));

   fBlueSlider = new TGHSlider(this, 100, kSlider1 | kScaleBoth, kHSb);
   fBlueSlider->Connect("PositionChanged(Int_t)", "TGLColorEditor", this, "DoSlider(Int_t)");
   fBlueSlider->SetRange(0, 100);
   fBlueSlider->SetPosition(Int_t(fRGBA[2] * 100));

   fAlphaSlider = new TGHSlider(this, 100, kSlider1 | kScaleBoth, kHSa);
   fAlphaSlider->Connect("PositionChanged(Int_t)", "TGLColorEditor", this, "DoSlider(Int_t)");
   fAlphaSlider->SetRange(0, 100);
   fAlphaSlider->SetPosition(Int_t(fRGBA[3] * 100));

   fShineSlider = new TGHSlider(this, 100, kSlider1 | kScaleBoth, kHSs);
   fShineSlider->Connect("PositionChanged(Int_t)", "TGLColorEditor", this, "DoSlider(Int_t)");
   fShineSlider->SetRange(0, 128);

   AddFrame(new TGLabel(this, "Red :"), new TGLayoutHints(kLHintsTop | kLHintsLeft, 5, 0, 0, 0));
   AddFrame(fRedSlider, new TGLayoutHints(kLHintsTop | kLHintsLeft | kLHintsExpandX, 0, 0, 0, 0));
   AddFrame(new TGLabel(this, "Green :"), new TGLayoutHints(kLHintsTop | kLHintsLeft, 5, 0, 0, 0));
   AddFrame(fGreenSlider, new TGLayoutHints(kLHintsTop | kLHintsLeft | kLHintsExpandX, 0, 0, 0, 0));
   AddFrame(new TGLabel(this, "Blue :"), new TGLayoutHints(kLHintsTop | kLHintsLeft, 5, 0, 0, 0));
   AddFrame(fBlueSlider, new TGLayoutHints(kLHintsTop | kLHintsLeft | kLHintsExpandX, 0, 0, 0, 0));
   AddFrame(new TGLabel(this, "Opacity :"), new TGLayoutHints(kLHintsTop | kLHintsLeft, 5, 0, 0, 0));
   AddFrame(fAlphaSlider, new TGLayoutHints(kLHintsTop | kLHintsLeft | kLHintsExpandX, 0, 0, 0, 0));
   AddFrame(new TGLabel(this, "Shine :"), new TGLayoutHints(kLHintsTop | kLHintsLeft, 5, 0, 0, 0));
   AddFrame(fShineSlider, new TGLayoutHints(kLHintsTop | kLHintsLeft | kLHintsExpandX, 0, 0, 0, 0));
}

//______________________________________________________________________________
void TGLColorEditor::SetSlidersPos()
{
   // Update GUI sliders from internal data
   fRedSlider->SetPosition(Int_t(fRGBA[fLMode * 4] * 100));
   fGreenSlider->SetPosition(Int_t(fRGBA[fLMode * 4 + 1] * 100));
   fBlueSlider->SetPosition(Int_t(fRGBA[fLMode * 4 + 2] * 100));
   fAlphaSlider->SetPosition(Int_t(fRGBA[fLMode * 4 + 3] * 100));

   if (fRGBA[16] >= 0.f)
      fShineSlider->SetPosition(Int_t(fRGBA[16]));
}

//______________________________________________________________________________
Bool_t TGLColorEditor::HandleContainerNotify(Event_t *event)
{
   // Handle resize event
   gVirtualX->ResizeWindow(fGLWin, event->fWidth, event->fHeight);
   DrawSphere();
   return kTRUE;
}

//______________________________________________________________________________
Bool_t TGLColorEditor::HandleContainerExpose(Event_t * /*event*/)
{
   // Handle expose (show) event
   DrawSphere();
   return kTRUE;
}

//______________________________________________________________________________
void TGLColorEditor::DrawSphere()const
{
   // Draw local sphere reflecting current color options
   MakeCurrent();
   gVirtualGL->ClearGL(0);
   if (fIsActive) {
      gVirtualGL->ViewportGL(0, 0, fMatView->GetWidth(), fMatView->GetHeight());
      gVirtualGL->NewMVGL();
      Float_t ligPos[] = {0.f, 0.f, 0.f, 1.f};
      gVirtualGL->GLLight(kLIGHT0, kPOSITION, ligPos);
      gVirtualGL->TranslateGL(0., 0., -3.);
      gVirtualGL->DrawSphere(fRGBA);
   }
   SwapBuffers();
}

//______________________________________________________________________________
void TGLColorEditor::MakeCurrent()const
{
   // Make our GL context current
   gVirtualGL->MakeCurrent(fGLWin, fCtx);
}

//______________________________________________________________________________
void TGLColorEditor::SwapBuffers()const
{
   // Swap our GL buffers
   gVirtualGL->SwapBuffers(fGLWin);
}

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGLGeometryEditor                                                    //
//                                                                      //
// GL Viewer shape geometry editor GUI component                        //
//////////////////////////////////////////////////////////////////////////

ClassImp(TGLGeometryEditor)

//______________________________________________________________________________
TGLGeometryEditor::TGLGeometryEditor(const TGWindow *parent, TGLSAViewer *v)
                     :TGCompositeFrame(parent, 100, 100, kVerticalFrame),// | kRaisedFrame),
                      fViewer(v), fGeomData(), fApplyButton(0), fIsActive(kFALSE)
{
   // Construct geometry editor GUI component, parented by window 'parent',
   // bound to viewer 'v'
   CreateCenterControls();
   CreateStretchControls();
   //create button
   fApplyButton = new TGTextButton(this, "Modify object", kTBa1);
   AddFrame(fApplyButton, new TGLayoutHints(kLHintsTop | kLHintsCenterX | kLHintsExpandX, 3, 3, 3, 3));
   fApplyButton->SetState(kButtonDisabled);
   fApplyButton->Connect("Pressed()", "TGLGeometryEditor", this, "DoButton()");
}

//______________________________________________________________________________
TGLGeometryEditor::~TGLGeometryEditor()
{
}

//______________________________________________________________________________
void TGLGeometryEditor::SetCenter(const Double_t *c)
{
   // Set internal center data from 3 component 'c'
   fIsActive = kTRUE;
   fApplyButton->SetState(kButtonDisabled);
   fGeomData[kCenterX]->SetNumber(c[0]);
   fGeomData[kCenterY]->SetNumber(c[1]);
   fGeomData[kCenterZ]->SetNumber(c[2]);
}

//______________________________________________________________________________
void TGLGeometryEditor::SetScale(const Double_t *s)
{
   // Set internal scale data from 3 component 'c'
   fIsActive = kTRUE;
   fGeomData[kScaleX]->SetNumber(s[0]);
   fGeomData[kScaleY]->SetNumber(s[1]);
   fGeomData[kScaleZ]->SetNumber(s[2]);
}

//______________________________________________________________________________
void TGLGeometryEditor::Disable()
{
   //Disable "Apply" button
   fIsActive = kFALSE;
   fApplyButton->SetState(kButtonDisabled);
}

//______________________________________________________________________________
void TGLGeometryEditor::DoButton()
{
   // Process 'Apply' - update the viewer object from GUI
   if (TGButton *btn = (TGButton *)gTQSender) {
      Int_t wid = btn->WidgetId();
      fViewer->ProcessGUIEvent(wid);
      if (wid == kTBa1) {
         fApplyButton->SetState(kButtonDisabled);
      }
   }
}

//______________________________________________________________________________
void TGLGeometryEditor::GetObjectData(Double_t *center, Double_t *scale)
{
   // Extract the GUI object data, return center in 3 component 'center'
   // scale in 3 component 'scale'
   center[0] = fGeomData[kCenterX]->GetNumber();
   center[1] = fGeomData[kCenterY]->GetNumber();
   center[2] = fGeomData[kCenterZ]->GetNumber();
   scale[0] = fGeomData[kScaleX]->GetNumber();
   scale[1] = fGeomData[kScaleY]->GetNumber();
   scale[2] = fGeomData[kScaleZ]->GetNumber();
}

//______________________________________________________________________________
void TGLGeometryEditor::ValueSet(Long_t)
{
   // Process setting of value in edit box - activate 'Apply' button
   if (!fIsActive)return;
   fApplyButton->SetState(kButtonUp);
}

//______________________________________________________________________________
void TGLGeometryEditor::CreateCenterControls()
{
   // Create object center GUI
   AddFrame(new TGLabel(this, "Object's center, X:"), new TGLayoutHints(kLHintsTop | kLHintsLeft, 3, 3, 3, 3));
   fGeomData[kCenterX] = new TGNumberEntry(this, 0.0, 8, kNExc);
   AddFrame(fGeomData[kCenterX], new TGLayoutHints(kLHintsTop | kLHintsCenterX | kLHintsExpandX, 3, 3, 3, 3));
   fGeomData[kCenterX]->Connect("ValueSet(Long_t)", "TGLGeometryEditor",
                                this, "ValueSet(Long_t)");

   AddFrame(new TGLabel(this, "Object's center, Y:"), new TGLayoutHints(kLHintsTop | kLHintsLeft, 3, 3, 3, 3));
   fGeomData[kCenterY] = new TGNumberEntry(this, 0.0, 8, kNEyc);
   AddFrame(fGeomData[kCenterY], new TGLayoutHints(kLHintsTop | kLHintsCenterX | kLHintsExpandX, 3, 3, 3, 3));
   fGeomData[kCenterY]->Connect("ValueSet(Long_t)", "TGLGeometryEditor",
                                this, "ValueSet(Long_t)");

   AddFrame(new TGLabel(this, "Object's center, Z:"), new TGLayoutHints(kLHintsTop | kLHintsLeft, 3, 3, 3, 3));
   fGeomData[kCenterZ] = new TGNumberEntry(this, 0.0, 8, kNEzc);
   AddFrame(fGeomData[kCenterZ], new TGLayoutHints(kLHintsTop | kLHintsCenterX | kLHintsExpandX, 3, 3, 3, 3));
   fGeomData[kCenterZ]->Connect("ValueSet(Long_t)", "TGLGeometryEditor",
                                this, "ValueSet(Long_t)");
}

//______________________________________________________________________________
void TGLGeometryEditor::CreateStretchControls()
{
   // Create object scale GUI
   AddFrame(new TGLabel(this, "Object's scale, X:"), new TGLayoutHints(kLHintsTop | kLHintsLeft, 3, 3, 3, 3));
   fGeomData[kScaleX] = new TGNumberEntry(this, 1.0, 8, kNExs);
   AddFrame(fGeomData[kScaleX], new TGLayoutHints(kLHintsTop | kLHintsCenterX | kLHintsExpandX, 3, 3, 3, 3));
   fGeomData[kScaleX]->Connect("ValueSet(Long_t)", "TGLGeometryEditor",
                               this, "ValueSet(Long_t)");

   AddFrame(new TGLabel(this, "Object's scale, Y:"), new TGLayoutHints(kLHintsTop | kLHintsLeft, 3, 3, 3, 3));
   fGeomData[kScaleY] = new TGNumberEntry(this, 1.0, 8, kNEys);
   AddFrame(fGeomData[kScaleY], new TGLayoutHints(kLHintsTop | kLHintsCenterX | kLHintsExpandX, 3, 3, 3, 3));
   fGeomData[kScaleY]->Connect("ValueSet(Long_t)", "TGLGeometryEditor",
                               this, "ValueSet(Long_t)");

   AddFrame(new TGLabel(this, "Object's scale, Z:"), new TGLayoutHints(kLHintsTop | kLHintsLeft, 3, 3, 3, 3));
   fGeomData[kScaleZ] = new TGNumberEntry(this, 1.0, 8, kNEzs);
   AddFrame(fGeomData[kScaleZ], new TGLayoutHints(kLHintsTop | kLHintsCenterX | kLHintsExpandX, 3, 3, 3, 3));
   fGeomData[kScaleZ]->Connect("ValueSet(Long_t)", "TGLGeometryEditor",
                               this, "ValueSet(Long_t)");

   fGeomData[kScaleX]->SetLimits(TGNumberFormat::kNELLimitMin, 0.1);
   fGeomData[kScaleY]->SetLimits(TGNumberFormat::kNELLimitMin, 0.1);
   fGeomData[kScaleZ]->SetLimits(TGNumberFormat::kNELLimitMin, 0.1);
}

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGLClipEditor                                                        //
//                                                                      //
// GL Viewer clipping shape editor GUI component                        //
//////////////////////////////////////////////////////////////////////////

ClassImp(TGLClipEditor)

//______________________________________________________________________________
TGLClipEditor::TGLClipEditor(const TGWindow *parent, TGLSAViewer *v) :
   TGCompositeFrame(parent, 100, 100, kVerticalFrame),
                    fViewer(v), fCurrentClip(kClipNone)
{
   // Construct clip editor GUI component, parented by window 'parent',
   // bound to viewer 'v'
   fTrash.SetOwner(kTRUE);
   fL1 = new TGLayoutHints(kLHintsTop | kLHintsCenterX | kLHintsExpandX, 3, 3, 3, 3);
   fTrash.AddLast(fL1);
   fL2 = new TGLayoutHints(kLHintsTop | kLHintsLeft, 3, 3, 3, 3);
   fTrash.AddLast(fL2);
   CreateControls();
}

//______________________________________________________________________________
TGLClipEditor::~TGLClipEditor()
{
}

//______________________________________________________________________________
void TGLClipEditor::CreateControls()
{
   // Create GUI controls - clip tyep (none/plane/box) and plane/box properties
   fTypeButtons = new TGButtonGroup(this, "Clip Type");
   fTrash.AddLast(fTypeButtons);
   TGRadioButton * clipNone = new TGRadioButton(fTypeButtons, "None");
   fTrash.AddLast(clipNone);
   TGRadioButton * clipPlane = new TGRadioButton(fTypeButtons, "Plane");
   fTrash.AddLast(clipPlane);
   TGRadioButton * clipBox = new TGRadioButton(fTypeButtons, "Box");
   fTrash.AddLast(clipBox);
   AddFrame(fTypeButtons, fL1);
   fTypeButtons->Connect("Pressed(Int_t)", "TGLClipEditor", this, "ClipTypeChanged(Int_t)");

   // Viewer Edit
   fEdit = new TGCheckButton(this, "Show / Edit In Viewer", kTBda);
   fTrash.AddLast(fEdit);
   AddFrame(fEdit, fL1);
   fEdit->Connect("Clicked()", "TGLClipEditor", this, "UpdateViewer()");

   // Plane properties
   fPlanePropFrame = new TGCompositeFrame(this);
   fTrash.AddLast(fPlanePropFrame);
   AddFrame(fPlanePropFrame, fL1);
   TGLabel * label;
   std::string planeStr[4] = { "aX + ", "bY +", "cZ + ", "d = 0" };
   UInt_t i;
   for (i=0; i<4; i++) {
      label = new TGLabel(fPlanePropFrame, planeStr[i].c_str());
      fTrash.AddLast(label);
      fPlanePropFrame->AddFrame(label, fL2);
      fPlaneProp[i] = new TGNumberEntry(fPlanePropFrame, 1., 8);
      fTrash.AddLast(fPlaneProp[i]);
      fPlanePropFrame->AddFrame(fPlaneProp[i], fL1);
      fPlaneProp[i]->Connect("ValueSet(Long_t)", "TGLClipEditor",
                             this, "ClipValueChanged(Long_t)");
   }

   // Box properties
   fBoxPropFrame = new TGCompositeFrame(this);
   fTrash.AddLast(fBoxPropFrame);
   AddFrame(fBoxPropFrame, fL1);

   std::string boxStr[6] = { "Center X", "Center Y", "Center Y", "Length X", "Length Y", "Length Z" };
   for (i=0; i<6; i++) {
      label = new TGLabel(fBoxPropFrame, boxStr[i].c_str());
      fTrash.AddLast(label);
      fBoxPropFrame->AddFrame(label, fL2);
      fBoxProp[i] = new TGNumberEntry(fBoxPropFrame, 1., 8);
      fTrash.AddLast(fBoxProp[i]);
      fBoxPropFrame->AddFrame(fBoxProp[i], fL1);
      fBoxProp[i]->Connect("ValueSet(Long_t)", "TGLClipEditor",
                           this, "ClipValueChanged(Long_t)");
   }

   // Apply button
   fApplyButton = new TGTextButton(this, "Apply", kTBcpm);
   fTrash.AddLast(fApplyButton);
   AddFrame(fApplyButton, fL1);
   fApplyButton->SetState(kButtonDisabled);
   fApplyButton->Connect("Pressed()", "TGLClipEditor", this, "UpdateViewer()");

   clipNone->SetState(kButtonDown);
}

//______________________________________________________________________________
void TGLClipEditor::HideParts()
{
   // Hide plane/box panels
   HideFrame(fPlanePropFrame);
   HideFrame(fBoxPropFrame);
}

//______________________________________________________________________________
void TGLClipEditor::ClipValueChanged(Long_t)
{
   // GUI value change - activate 'Apply' button
   fApplyButton->SetState(kButtonUp);
}

//______________________________________________________________________________
void TGLClipEditor::ClipTypeChanged(Int_t id)
{
   // Clip type radio button changed - update viewer
   if (id == 1) {
      SetCurrent(kClipNone, kFALSE);
      fEdit->SetState(kButtonDisabled);
   } else {
      if (fEdit->GetState() == kButtonDisabled) {
         fEdit->SetState(kButtonUp);
      }
      SetCurrent(id == 2 ? kClipPlane : kClipBox, fEdit->IsDown());
   }

   // Internal GUI change - need to update the viewer
   UpdateViewer();
}

//______________________________________________________________________________
void TGLClipEditor::UpdateViewer()
{
   // Update viewer for GUI change
   fViewer->ProcessGUIEvent(kTBcpm);
   fApplyButton->SetState(kButtonDisabled);
}

//______________________________________________________________________________
void TGLClipEditor::GetState(EClipType type, Double_t data[6]) const
{
   // Fetch GUI state for clip if 'type' into 'data' vector
   UInt_t i;
   if (type == kClipNone) {
      // Nothing to do
   } else if (type == kClipPlane) {
      for (i=0; i<4; i++) {
         data[i] = fPlaneProp[i]->GetNumber();
      }
   } else if (type == kClipBox) {
      for (i=0; i<6; i++) {
         data[i] = fBoxProp[i]->GetNumber();
      }
   } else {
      Error("TGLClipEditor::GetClipState", "Invalid clip type");
   }
}

//______________________________________________________________________________
void TGLClipEditor::SetState(EClipType type, const Double_t data[6])
{
   // Set GUI state for clip 'type from 'data' vector
   UInt_t i;
   if (type == kClipNone) {
      // Nothing to do
   } else if (type == kClipPlane) {
      for (i=0; i<4; i++) {
         fPlaneProp[i]->SetNumber(data[i]);
      }
   } else if (type == kClipBox) {
      for (i=0; i<6; i++) {
         fBoxProp[i]->SetNumber(data[i]);
      }
   } else {
      Error("TGLClipEditor::SetClipState", "Invalid clip type");
   }
   fApplyButton->SetState(kButtonDisabled);
}

//______________________________________________________________________________
void TGLClipEditor::GetCurrent(EClipType & type, Bool_t & edit) const
{
   // Get current (active) GUI clip type into 'type', and in viewer edit
   // state into 'edit'
   type = fCurrentClip;
   edit = fEdit->IsDown();
}

//______________________________________________________________________________
void TGLClipEditor::SetCurrent(EClipType type, Bool_t edit)
{
   // Set current (active) GUI clip type from 'type'
   fCurrentClip = type;
   switch(fCurrentClip) {
      case(kClipNone): {
         fTypeButtons->SetButton(1);
         HideFrame(fPlanePropFrame);
         HideFrame(fBoxPropFrame);
         break;
      }
      case(kClipPlane): {
         fTypeButtons->SetButton(2);
         ShowFrame(fPlanePropFrame);
         HideFrame(fBoxPropFrame);
         break;
      }
      case(kClipBox): {
         fTypeButtons->SetButton(3);
         HideFrame(fPlanePropFrame);
         ShowFrame(fBoxPropFrame);
         break;
      }
      default: {
         Error("TGLClipEditor::SetCurrentClip", "Invalid clip type");
         break;
      }
   }
   fEdit->SetDown(edit);
}

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGLLightEditor                                                       //
//                                                                      //
// GL Viewer lighting editor GUI component                              //
//////////////////////////////////////////////////////////////////////////

ClassImp(TGLLightEditor)

//______________________________________________________________________________
TGLLightEditor::TGLLightEditor(const TGWindow *parent, TGLSAViewer *v)
               :TGCompositeFrame(parent, 100, 100, kVerticalFrame),// | kRaisedFrame),
                fViewer(v)
{
   // Construct light editor GUI component, parented by window 'parent',
   // bound to viewer 'v'
   fTrash.SetOwner(kTRUE);
   TGGroupFrame *ligFrame = new TGGroupFrame(this, "Sources", kLHintsTop | kLHintsCenterX);
   fTrash.AddLast(ligFrame);
   ligFrame->SetTitlePos(TGGroupFrame::kLeft);
   TGLayoutHints *l = new TGLayoutHints(kLHintsTop | kLHintsCenterX | kLHintsExpandX, 3, 3, 3, 3);
   fTrash.AddLast(l);
   AddFrame(ligFrame, l);

   TGMatrixLayout *ml = new TGMatrixLayout(ligFrame, 0, 1, 10);
   ligFrame->SetLayoutManager(ml);
   // ligFrame will delete the layout manager ml for us so don't add to fTrash

   fLights[kTop] = new TGCheckButton(ligFrame, "Top", kTBTop);
   fLights[kTop]->Connect("Clicked()", "TGLLightEditor", this, "DoButton()");
   fLights[kTop]->SetState(kButtonDown);
   fTrash.AddLast(fLights[kTop]);
   fLights[kRight] = new TGCheckButton(ligFrame, "Right", kTBRight);
   fLights[kRight]->Connect("Clicked()", "TGLLightEditor", this, "DoButton()");
   fLights[kRight]->SetState(kButtonDown);
   fTrash.AddLast(fLights[kRight]);
   fLights[kBottom] = new TGCheckButton(ligFrame, "Bottom", kTBBottom);
   fLights[kBottom]->Connect("Clicked()", "TGLLightEditor", this, "DoButton()");
   fLights[kBottom]->SetState(kButtonDown);
   fTrash.AddLast(fLights[kBottom]);
   fLights[kLeft] = new TGCheckButton(ligFrame, "Left", kTBLeft);
   fLights[kLeft]->Connect("Clicked()", "TGLLightEditor", this, "DoButton()");
   fLights[kLeft]->SetState(kButtonDown);
   fTrash.AddLast(fLights[kLeft]);
   fLights[kFront] = new TGCheckButton(ligFrame, "Front", kTBFront);
   fLights[kFront]->Connect("Clicked()", "TGLLightEditor", this, "DoButton()");
   fLights[kFront]->SetState(kButtonDown);
   fTrash.AddLast(fLights[kFront]);

   ligFrame->AddFrame(fLights[kTop]);
   ligFrame->AddFrame(fLights[kRight]);
   ligFrame->AddFrame(fLights[kBottom]);
   ligFrame->AddFrame(fLights[kLeft]);
   ligFrame->AddFrame(fLights[kFront]);
}

//______________________________________________________________________________
TGLLightEditor::~TGLLightEditor()
{
}

//______________________________________________________________________________
void TGLLightEditor::DoButton()
{
   // Process light GUI change - send to viewer
   TGButton *btn = (TGButton *) gTQSender;
   Int_t id = btn->WidgetId();
   fViewer->ProcessGUIEvent(id);
}

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGLGuideEditor                                                       //
//                                                                      //
// GL Viewer guides editor GUI component                                //
//////////////////////////////////////////////////////////////////////////

ClassImp(TGLGuideEditor)

//______________________________________________________________________________
TGLGuideEditor::TGLGuideEditor(const TGWindow *parent, TGLSAViewer *v) :
   TGCompositeFrame(parent, 100, 100, kVerticalFrame),
   fViewer(v),  fAxesContainer(0), fReferenceContainer(0),
   fReferenceOn(0),
   fL1(0), fL2(0)
{
   // Construct guide editor GUI component, parented by window 'parent',
   // bound to viewer 'v'
   fTrash.SetOwner(kTRUE);

   fL1 = new TGLayoutHints(kLHintsTop | kLHintsCenterX | kLHintsExpandX, 0, 0, 1, 1);
   fTrash.AddLast(fL1);
   fL2 = new TGLayoutHints(kLHintsTop | kLHintsLeft, 0, 0, 3, 3);
   fTrash.AddLast(fL2);

   // Axes container
   fAxesContainer = new TGButtonGroup(this, "Axes");
   fTrash.AddLast(fAxesContainer);
   AddFrame(fAxesContainer, fL1);
   fAxesContainer->Connect("Pressed(Int_t)", "TGLGuideEditor", this, "Update()");

   // Axes options
   TGRadioButton * axesNone = new TGRadioButton(fAxesContainer, "None");
   fTrash.AddLast(axesNone);
   TGRadioButton * axesEdge = new TGRadioButton(fAxesContainer, "Edge");
   fTrash.AddLast(axesEdge);
   TGRadioButton * axesOrigins = new TGRadioButton(fAxesContainer, "Origin");
   fTrash.AddLast(axesOrigins);

   // Reference container
   fReferenceContainer = new TGGroupFrame(this, "Reference Marker");
   fTrash.AddLast(fReferenceContainer);
   AddFrame(fReferenceContainer, fL1);

   // Reference options
   fReferenceOn = new TGCheckButton(fReferenceContainer, "Show");
   fTrash.AddLast(fReferenceOn);
   fReferenceContainer->AddFrame(fReferenceOn, fL1);
   fReferenceOn->Connect("Clicked()", "TGLGuideEditor", this, "Update()");

   TGLabel * label = new TGLabel(fReferenceContainer, "X");
   fTrash.AddLast(label);
   fReferenceContainer->AddFrame(label, fL2);
   fReferencePos[0] = new TGNumberEntry(fReferenceContainer, 0.0, 8);
   fTrash.AddLast(fReferencePos[0]);
   fReferenceContainer->AddFrame(fReferencePos[0], fL1);
   fReferencePos[0]->Connect("ValueSet(Long_t)", "TGLGuideEditor",
                             this, "Update()");

   label = new TGLabel(fReferenceContainer, "Y");
   fTrash.AddLast(label);
   fReferenceContainer->AddFrame(label, fL2);
   fReferencePos[1] = new TGNumberEntry(fReferenceContainer, 0.0, 8);
   fTrash.AddLast(fReferencePos[1]);
   fReferenceContainer->AddFrame(fReferencePos[1], fL1);
   fReferencePos[1]->Connect("ValueSet(Long_t)", "TGLGuideEditor",
                             this, "Update()");

   label = new TGLabel(fReferenceContainer, "Z");
   fTrash.AddLast(label);
   fReferenceContainer->AddFrame(label, fL2);
   fReferencePos[2] = new TGNumberEntry(fReferenceContainer, 0.0, 8);
   fTrash.AddLast(fReferencePos[2]);
   fReferenceContainer->AddFrame(fReferencePos[2], fL1);
   fReferencePos[2]->Connect("ValueSet(Long_t)", "TGLGuideEditor",
                             this, "Update()");
   axesNone->SetState(kButtonDown);
}

//______________________________________________________________________________
TGLGuideEditor::~TGLGuideEditor()
{
}

//______________________________________________________________________________
void TGLGuideEditor::Update()
{
   // Update viewer with GUI state
   fViewer->ProcessGUIEvent(kTBGuide);
   UpdateReferencePos();
}

//______________________________________________________________________________
void TGLGuideEditor::GetState(TGLViewer::EAxesType & axesType, Bool_t & referenceOn, Double_t referencePos[3]) const
{
   // Get GUI state into arguments:
   // 'axesType'     - axes type - one of EAxesType - kAxesNone/kAxesPlane/kAxesBox
   // 'referenceOn'  - reference marker on (visible)
   // 'referencePos' - current reference position (vertex)

   // Button ids run from 1
   for (Int_t i = 1; i < 4; i++) {
      TGButton * button = fAxesContainer->GetButton(i);
      if (button && button->IsDown()) {
         axesType = TGLViewer::EAxesType(i-1);
      }
   }
   referenceOn = fReferenceOn->IsDown();
   referencePos[0] = fReferencePos[0]->GetNumber();
   referencePos[1] = fReferencePos[1]->GetNumber();
   referencePos[2] = fReferencePos[2]->GetNumber();
}

//______________________________________________________________________________
void TGLGuideEditor::SetState(TGLViewer::EAxesType axesType, Bool_t referenceOn, const Double_t referencePos[3])
{
   // Set GUI state from arguments:
   // 'axesType'     - axes type - one of EAxesType - kAxesNone/kAxesPlane/kAxesBox
   // 'referenceOn'  - reference marker on (visible)
   // 'referencePos' - current reference position (vertex)

   // Button ids run from 1
   TGButton * button = fAxesContainer->GetButton(axesType+1);
   if (button) {
      button->SetDown();
   }
   fReferenceOn->SetDown(referenceOn);
   fReferencePos[0]->SetNumber(referencePos[0]);
   fReferencePos[1]->SetNumber(referencePos[1]);
   fReferencePos[2]->SetNumber(referencePos[2]);
   UpdateReferencePos();
}

//______________________________________________________________________________
void TGLGuideEditor::UpdateReferencePos()
{
   // Enable/disable reference position (x/y/z) number edits based on
   // reference check box
   fReferencePos[0]->SetState(fReferenceOn->IsDown());
   fReferencePos[1]->SetState(fReferenceOn->IsDown());
   fReferencePos[2]->SetState(fReferenceOn->IsDown());
}
