// @(#)root/gl:$Name:  $:$Id: TGLColorEditor.cxx,v 1.5 2004/09/29 06:55:13 brun Exp $
// Author:  Timur Pocheptsov  03/08/2004

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/
#include <iostream>

#include "TVirtualGL.h"
#include "TVirtualX.h"
#include "TGCanvas.h"
#include "TGLayout.h"
#include "TGButton.h"
#include "TGSlider.h"
#include "TGLabel.h"
#include "TGNumberEntry.h"
#include "TViewerOpenGL.h"

#include "TGLEditor.h"

ClassImp(TGLColorEditor)

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
   AddInput(kExposureMask | kStructureNotifyMask);
}

//______________________________________________________________________________
Bool_t TGLMatView::HandleConfigureNotify(Event_t *event)
{
   return fOwner->HandleContainerNotify(event);
}

//______________________________________________________________________________
Bool_t TGLMatView::HandleExpose(Event_t *event)
{
   return fOwner->HandleContainerExpose(event);
}

enum EGLEditorIdent{
   kCPa = kTBa + 1,
   kCPd,
   kCPs,
   kCPe,
   kHSr,
   kHSg,
   kHSb,
   kHSa,
   kHSs,
   kHSe,
   kNExc,
   kNEyc,
   kNEzc,
   kNExs,
   kNEys,
   kNEzs,
};

//______________________________________________________________________________
TGLColorEditor::TGLColorEditor(const TGWindow *parent, TGWindow *main)
               :TGCompositeFrame(parent, 100, 100, kVerticalFrame | kRaisedFrame),
                fRedSlider(0), fGreenSlider(0), fBlueSlider(0), fAlphaSlider(0),
                fApplyButton(0), fIsActive(kFALSE), fIsLight(kFALSE), fRGBA()
{
   fTrash.SetOwner(kTRUE);

   for (Int_t i = 0; i < 12; ++i) fRGBA[i] = 1.;
   fRGBA[12] = fRGBA[13] = fRGBA[14] = 0.f;
   fRGBA[15] = 1.f, fRGBA[16] = 60.f;

   //Small gl-window with sphere
   TGCanvas *viewCanvas = new TGCanvas(this, 120, 120, kSunkenFrame | kDoubleBorder);
   fTrash.Add(viewCanvas);
   Window_t wid = viewCanvas->GetViewPort()->GetId();
   fGLWin = gVirtualGL->CreateGLWindow(wid);
   fMatView = new TGLMatView(viewCanvas->GetViewPort(), fGLWin, this);
   fTrash.Add(fMatView);
   fCtx = gVirtualGL->CreateContext(fGLWin);
   viewCanvas->SetContainer(fMatView);
   fFrameLayout = new TGLayoutHints(kLHintsTop | kLHintsCenterX, 2, 0, 2, 2);
   fTrash.Add(fFrameLayout);
   AddFrame(viewCanvas, fFrameLayout);

   CreateRadioButtons();
   fLMode = kDiffuse;
   fLightTypes[fLMode]->SetState(kButtonDown);

   CreateSliders();
   //apply button creation
   TGLayoutHints *widLayout = new TGLayoutHints(kLHintsTop | kLHintsLeft | kLHintsExpandX, 2, 2, 5, 0);
   fTrash.Add(widLayout);
   fApplyButton = new TGTextButton(this, "Apply", kTBa);
   fTrash.Add(fApplyButton);
   AddFrame(fApplyButton, widLayout);
   fApplyButton->SetState(kButtonDisabled);
   fApplyButton->Connect("Pressed()", "TViewerOpenGL", (TViewerOpenGL *)main, "ModifySelected()");
   fApplyButton->Connect("Pressed()", "TGLColorEditor", this, "DoButton()");

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
   gVirtualGL->DeleteContext(fCtx);
}

//______________________________________________________________________________
void TGLColorEditor::SetRGBA(const Float_t *rgba)
{
   fApplyButton->SetState(kButtonDisabled);
   fIsActive = kTRUE;
   for (Int_t i = 0; i < 17; ++i) fRGBA[i] = rgba[i];

   if (rgba[16] < 0.f) {
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
         if (!fIsLight) fRGBA[3] = val / 100.f;
         break;
      case kHSs:
         if (!fIsLight) fRGBA[16] = val;
         break;
      }
      if (!fIsLight || (wid != kHSa && wid != kHSs)) {
         if (fIsActive) fApplyButton->SetState(kButtonUp);
         DrawSphere();
      }
   }
}

//______________________________________________________________________________
void TGLColorEditor::DoButton()
{
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
      fApplyButton->SetState(kButtonDisabled);
      break;
   }
   DrawSphere();
}

//______________________________________________________________________________
void TGLColorEditor::Disable()
{
   fApplyButton->SetState(kButtonDisabled);
   fIsActive = kFALSE;
   fIsLight = kFALSE;
}

//______________________________________________________________________________
void TGLColorEditor::CreateRadioButtons()
{
   TGGroupFrame *partFrame = new TGGroupFrame(this, "Light:", kLHintsTop | kLHintsCenterX);
   fTrash.Add(partFrame);
   partFrame->SetTitlePos(TGGroupFrame::kLeft);
   AddFrame(partFrame, fFrameLayout);
   TGMatrixLayout *ml = new TGMatrixLayout(partFrame, 0, 1, 10);
   fTrash.Add(ml);
   partFrame->SetLayoutManager(ml);

   fLightTypes[kDiffuse] = new TGRadioButton(partFrame, "Diffuse color", kCPd);
   fLightTypes[kDiffuse]->Connect("Pressed()", "TGLColorEditor", this, "DoButton()");
   fTrash.Add(fLightTypes[kDiffuse]);
   fLightTypes[kAmbient] = new TGRadioButton(partFrame, "Ambient color", kCPa);
   fLightTypes[kAmbient]->Connect("Pressed()", "TGLColorEditor", this, "DoButton()");
   fTrash.Add(fLightTypes[kAmbient]);
   fLightTypes[kSpecular] = new TGRadioButton(partFrame, "Specular color", kCPs);
   fLightTypes[kSpecular]->Connect("Pressed()", "TGLColorEditor", this, "DoButton()");
   fTrash.Add(fLightTypes[kSpecular]);
   fLightTypes[kEmission] = new TGRadioButton(partFrame, "Emission color", kCPe);
   fLightTypes[kEmission]->Connect("Pressed()", "TGLColorEditor", this, "DoButton()");
   fTrash.Add(fLightTypes[kEmission]);

   partFrame->AddFrame(fLightTypes[kDiffuse]);
   partFrame->AddFrame(fLightTypes[kAmbient]);
   partFrame->AddFrame(fLightTypes[kSpecular]);
   partFrame->AddFrame(fLightTypes[kEmission]);
}

//______________________________________________________________________________
void TGLColorEditor::CreateSliders()
{
   fRedSlider = new TGHSlider(this, 100, kSlider1 | kScaleBoth, kHSr);
   fTrash.Add(fRedSlider);
   fRedSlider->Connect("PositionChanged(Int_t)", "TGLColorEditor", this, "DoSlider(Int_t)");
   fRedSlider->SetRange(0, 100);
   fRedSlider->SetPosition(Int_t(fRGBA[0] * 100));

   fGreenSlider = new TGHSlider(this, 100, kSlider1 | kScaleBoth, kHSg);
   fTrash.Add(fGreenSlider);
   fGreenSlider->Connect("PositionChanged(Int_t)", "TGLColorEditor", this, "DoSlider(Int_t)");
   fGreenSlider->SetRange(0, 100);
   fGreenSlider->SetPosition(Int_t(fRGBA[1] * 100));

   fBlueSlider = new TGHSlider(this, 100, kSlider1 | kScaleBoth, kHSb);
   fTrash.Add(fBlueSlider);
   fBlueSlider->Connect("PositionChanged(Int_t)", "TGLColorEditor", this, "DoSlider(Int_t)");
   fBlueSlider->SetRange(0, 100);
   fBlueSlider->SetPosition(Int_t(fRGBA[2] * 100));

   fAlphaSlider = new TGHSlider(this, 100, kSlider1 | kScaleBoth, kHSa);
   fTrash.Add(fAlphaSlider);
   fAlphaSlider->Connect("PositionChanged(Int_t)", "TGLColorEditor", this, "DoSlider(Int_t)");
   fAlphaSlider->SetRange(0, 100);
   fAlphaSlider->SetPosition(Int_t(fRGBA[3] * 100));

   fShineSlider = new TGHSlider(this, 100, kSlider1 | kScaleBoth, kHSs);
   fTrash.Add(fShineSlider);
   fShineSlider->Connect("PositionChanged(Int_t)", "TGLColorEditor", this, "DoSlider(Int_t)");
   fShineSlider->SetRange(0, 128);

   TGLabel *labelInfo[5] = {0};
   labelInfo[0] = new TGLabel(this, "Red :");
   fTrash.Add(labelInfo[0]);
   labelInfo[1] = new TGLabel(this, "Green :");
   fTrash.Add(labelInfo[1]);
   labelInfo[2] = new TGLabel(this, "Blue :");
   fTrash.Add(labelInfo[2]);
   labelInfo[3] = new TGLabel(this, "Opacity :");
   fTrash.Add(labelInfo[3]);
   labelInfo[4] = new TGLabel(this, "Shine :");
   fTrash.Add(labelInfo[4]);

   TGLayoutHints *layout1 = new TGLayoutHints(kLHintsTop | kLHintsLeft, 5, 0, 0, 0);
   fTrash.Add(layout1);
   TGLayoutHints *layout2 = new TGLayoutHints(kLHintsTop | kLHintsLeft | kLHintsExpandX, 0, 0, 0, 0);
   fTrash.Add(layout2);

   AddFrame(labelInfo[0], layout1);
   AddFrame(fRedSlider, layout2);
   AddFrame(labelInfo[1], layout1);
   AddFrame(fGreenSlider, layout2);
   AddFrame(labelInfo[2], layout1);
   AddFrame(fBlueSlider, layout2);
   AddFrame(labelInfo[3], layout1);
   AddFrame(fAlphaSlider, layout2);
   AddFrame(labelInfo[4], layout1);
   AddFrame(fShineSlider, layout2);
}

//______________________________________________________________________________
void TGLColorEditor::SetSlidersPos()
{
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
   gVirtualX->ResizeWindow(fGLWin, event->fWidth, event->fHeight);
   DrawSphere();
   return kTRUE;
}

//______________________________________________________________________________
Bool_t TGLColorEditor::HandleContainerExpose(Event_t * /*event*/)
{
   DrawSphere();
   return kTRUE;
}

//______________________________________________________________________________
void TGLColorEditor::DrawSphere()const
{
   MakeCurrent();
   gVirtualGL->ClearGL(0);
   gVirtualGL->ViewportGL(0, 0, fMatView->GetWidth(), fMatView->GetHeight());
   gVirtualGL->NewMVGL();
   Float_t ligPos[] = {0.f, 0.f, 0.f, 1.f};
   gVirtualGL->GLLight(kLIGHT0, kPOSITION, ligPos);
   gVirtualGL->TranslateGL(0., 0., -3.);
   gVirtualGL->DrawSphere(fRGBA);
   SwapBuffers();
}

//______________________________________________________________________________
void TGLColorEditor::MakeCurrent()const
{
   gVirtualGL->MakeCurrent(fGLWin, fCtx);
}

//______________________________________________________________________________
void TGLColorEditor::SwapBuffers()const
{
   gVirtualGL->SwapBuffers(fGLWin);
}

//______________________________________________________________________________
TGLGeometryEditor::TGLGeometryEditor(const TGWindow *parent, TGWindow *main)
                     :TGCompositeFrame(parent, 100, 100, kVerticalFrame | kRaisedFrame)
{
   fTrash.SetOwner(kTRUE);
   fIsActive = kFALSE;
   fFrameLayout = new TGLayoutHints(kLHintsTop | kLHintsCenterX | kLHintsExpandX, 1, 1, 1, 1);
   fTrash.AddLast(fFrameLayout);
   CreateCenterFrame();
   CreateScaleFrame();
   //create button
   fApplyButton = new TGTextButton(this, "Apply", kTBa1);
   fTrash.Add(fApplyButton);
   AddFrame(fApplyButton, fFrameLayout);
   fApplyButton->SetState(kButtonDisabled);
   fApplyButton->Connect("Pressed()", "TViewerOpenGL", (TViewerOpenGL *)main, "ModifySelected()");
   fApplyButton->Connect("Pressed()", "TGLGeometryEditor", this, "DoButton()");
}

//______________________________________________________________________________
void TGLGeometryEditor::SetCenter(const Double_t *center)
{
   fIsActive = kTRUE;
   fApplyButton->SetState(kButtonDisabled);
   fGeomData[kCenterX]->SetNumber(center[0]);
   fGeomData[kCenterY]->SetNumber(center[1]);
   fGeomData[kCenterZ]->SetNumber(center[2]);
   fGeomData[kScaleX]->SetNumber(1.0);
   fGeomData[kScaleY]->SetNumber(1.0);
   fGeomData[kScaleZ]->SetNumber(1.0);
}

//______________________________________________________________________________
void TGLGeometryEditor::Disable()
{
   fIsActive = kFALSE;
   fApplyButton->SetState(kButtonDisabled);
}

//______________________________________________________________________________
void TGLGeometryEditor::DoButton()
{
   fApplyButton->SetState(kButtonDisabled);
   fGeomData[kScaleX]->SetNumber(1.0);
   fGeomData[kScaleY]->SetNumber(1.0);
   fGeomData[kScaleZ]->SetNumber(1.0);
}

//______________________________________________________________________________
void TGLGeometryEditor::GetNewData(Double_t *center, Double_t *scale)
{
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
   if (!fIsActive) return;
   if (fApplyButton->GetState() == kButtonDisabled) fApplyButton->SetState(kButtonUp);
}

//______________________________________________________________________________
void TGLGeometryEditor::CreateCenterFrame()
{
   TGGroupFrame *groupFrame = new TGGroupFrame(this, "Object center", kLHintsTop | kLHintsCenterX);
   groupFrame->SetTitlePos(TGGroupFrame::kLeft);
   fTrash.AddLast(groupFrame);
   AddFrame(groupFrame, fFrameLayout);
   TGMatrixLayout *ml = new TGMatrixLayout(groupFrame, 0, 1, 10);
   fTrash.AddLast(ml);
   groupFrame->SetLayoutManager(ml);

   TGLabel *label = new TGLabel(groupFrame, "X :");
   fTrash.AddLast(label);
   groupFrame->AddFrame(label);  
   fGeomData[kCenterX] = new TGNumberEntry(groupFrame, 0.0, 8, kNExc);
   fTrash.AddLast(fGeomData[kCenterX]);
   groupFrame->AddFrame(fGeomData[kCenterX]);
   fGeomData[kCenterX]->Connect("ValueSet(Long_t)", "TGLGeometryEditor", this, "ValueSet(Long_t)");

   label = new TGLabel(groupFrame, "Y :");
   fTrash.AddLast(label);
   groupFrame->AddFrame(label);  
   fGeomData[kCenterY] = new TGNumberEntry(groupFrame, 0.0, 8, kNEyc);
   fTrash.AddLast(fGeomData[kCenterY]);
   groupFrame->AddFrame(fGeomData[kCenterY]);
   fGeomData[kCenterY]->Connect("ValueSet(Long_t)", "TGLGeometryEditor", this, "ValueSet(Long_t)");

   label = new TGLabel(groupFrame, "Z :");
   fTrash.AddLast(label);
   groupFrame->AddFrame(label);  
   fGeomData[kCenterZ] = new TGNumberEntry(groupFrame, 0.0, 8, kNEzc);
   fTrash.AddLast(fGeomData[kCenterZ]);
   groupFrame->AddFrame(fGeomData[kCenterZ]);
   fGeomData[kCenterZ]->Connect("ValueSet(Long_t)", "TGLGeometryEditor", this, "ValueSet(Long_t)");
}

//______________________________________________________________________________
void TGLGeometryEditor::CreateScaleFrame()
{
   TGGroupFrame *groupFrame = new TGGroupFrame(this, "Stretch", kLHintsTop | kLHintsCenterX);
   groupFrame->SetTitlePos(TGGroupFrame::kLeft);
   fTrash.AddLast(groupFrame);
   AddFrame(groupFrame, fFrameLayout);
   TGMatrixLayout *ml = new TGMatrixLayout(groupFrame, 0, 1, 10);
   fTrash.AddLast(ml);
   groupFrame->SetLayoutManager(ml);

   TGLabel *label = new TGLabel(groupFrame, "X :");
   fTrash.AddLast(label);
   groupFrame->AddFrame(label);  
   fGeomData[kScaleX] = new TGNumberEntry(groupFrame, 1.0, 8, kNExs);
   fTrash.AddLast(fGeomData[kScaleX]);
   groupFrame->AddFrame(fGeomData[kScaleX]);
   fGeomData[kScaleX]->Connect("ValueSet(Long_t)", "TGLGeometryEditor", this, "ValueSet(Long_t)");

   label = new TGLabel(groupFrame, "Y :");
   fTrash.AddLast(label);
   groupFrame->AddFrame(label);  
   fGeomData[kScaleY] = new TGNumberEntry(groupFrame, 1.0, 8, kNEys);
   fTrash.AddLast(fGeomData[kScaleY]);
   groupFrame->AddFrame(fGeomData[kScaleY]);
   fGeomData[kScaleY]->Connect("ValueSet(Long_t)", "TGLGeometryEditor", this, "ValueSet(Long_t)");

   label = new TGLabel(groupFrame, "Z :");
   fTrash.AddLast(label);
   groupFrame->AddFrame(label);  
   fGeomData[kScaleZ] = new TGNumberEntry(groupFrame, 1.0, 8, kNEzs);
   fTrash.AddLast(fGeomData[kScaleZ]);
   groupFrame->AddFrame(fGeomData[kScaleZ]);
   fGeomData[kScaleZ]->Connect("ValueSet(Long_t)", "TGLGeometryEditor", this, "ValueSet(Long_t)");

   fGeomData[kScaleX]->SetLimits(TGNumberFormat::kNELLimitMin, 0.1);
   fGeomData[kScaleY]->SetLimits(TGNumberFormat::kNELLimitMin, 0.1);
   fGeomData[kScaleZ]->SetLimits(TGNumberFormat::kNELLimitMin, 0.1);
}

