// @(#)root/gl:$Name:  $:$Id: TGLEditor.cxx,v 1.4 2004/09/17 11:47:16 rdm Exp $
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
#include "TGSlider.h"
#include "TGLabel.h"
#include "TViewerOpenGL.h"

#include "TGLEditor.h"

ClassImp(TGLEditor)

class TGLMatView : public TGCompositeFrame {
private:
   TGLEditor *fOwner;
public:
   TGLMatView(const TGWindow *parent, Window_t wid, TGLEditor *owner);
   Bool_t HandleConfigureNotify(Event_t *event);
   Bool_t HandleExpose(Event_t *event);

private:
   TGLMatView(const TGLMatView &);
   TGLMatView & operator = (const TGLMatView &);
};

//______________________________________________________________________________
TGLMatView::TGLMatView(const TGWindow *parent, Window_t wid, TGLEditor *owner)
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
   kCPa,
   kCPd,
   kCPs,
   kCPe,
   kHSr,
   kHSg,
   kHSb,
   kHSa,
   kHSs,
   kHSe,
   kTBa
};

//______________________________________________________________________________
TGLEditor::TGLEditor(const TGWindow *parent, TGWindow *main)
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

   MakeCurrent();
   gVirtualGL->NewPRGL();
   gVirtualGL->FrustumGL(-0.5, 0.5, -0.5, 0.5, 1., 10.);
   gVirtualGL->EnableGL(kLIGHTING);
   gVirtualGL->EnableGL(kLIGHT0);
   gVirtualGL->EnableGL(kDEPTH_TEST);
   gVirtualGL->EnableGL(kCULL_FACE);
   gVirtualGL->CullFaceGL(kBACK);
   fApplyButton->Connect("Pressed()", "TViewerOpenGL", (TViewerOpenGL *)main, "ModifySelected()");
   fApplyButton->Connect("Pressed()", "TGLEditor", this, "DoButton()");
   DrawSphere();
}

//______________________________________________________________________________
TGLEditor::~TGLEditor()
{
   gVirtualGL->DeleteContext(fCtx);
}

//______________________________________________________________________________
void TGLEditor::SetRGBA(const Float_t *rgba)
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
void TGLEditor::DoSlider(Int_t val)
{
   TGSlider *frm = (TGSlider *)gTQSender;

   if (frm) {
      Int_t wid = frm->WidgetId();

      switch (wid) {
      case kHSr:
         fRGBA[fLMode * 4] = val / 100.f;;
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
void TGLEditor::DoButton()
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
void TGLEditor::Stop()
{
   fApplyButton->SetState(kButtonDisabled);
   fIsActive = kFALSE;
   fIsLight = kFALSE;
}

//______________________________________________________________________________
void TGLEditor::CreateRadioButtons()
{
   fPartFrame = new TGGroupFrame(this, "Light:", kLHintsTop | kLHintsCenterX);
   fTrash.Add(fPartFrame);
   fPartFrame->SetTitlePos(TGGroupFrame::kLeft);
   AddFrame(fPartFrame, fFrameLayout);
   TGMatrixLayout *ml = new TGMatrixLayout(fPartFrame, 0, 1, 10);
   fTrash.Add(ml);
   fPartFrame->SetLayoutManager(ml);

   fLightTypes[kDiffuse] = new TGRadioButton(fPartFrame, "Diffuse color", kCPd);
   fLightTypes[kDiffuse]->Connect("Pressed()", "TGLEditor", this, "DoButton()");
   fTrash.Add(fLightTypes[kDiffuse]);
   fLightTypes[kAmbient] = new TGRadioButton(fPartFrame, "Ambient color", kCPa);
   fLightTypes[kAmbient]->Connect("Pressed()", "TGLEditor", this, "DoButton()");
   fTrash.Add(fLightTypes[kAmbient]);
   fLightTypes[kSpecular] = new TGRadioButton(fPartFrame, "Specular color", kCPs);
   fLightTypes[kSpecular]->Connect("Pressed()", "TGLEditor", this, "DoButton()");
   fTrash.Add(fLightTypes[kSpecular]);
   fLightTypes[kEmission] = new TGRadioButton(fPartFrame, "Emission color", kCPe);
   fLightTypes[kEmission]->Connect("Pressed()", "TGLEditor", this, "DoButton()");
   fTrash.Add(fLightTypes[kEmission]);

   fPartFrame->AddFrame(fLightTypes[kDiffuse]);
   fPartFrame->AddFrame(fLightTypes[kAmbient]);
   fPartFrame->AddFrame(fLightTypes[kSpecular]);
   fPartFrame->AddFrame(fLightTypes[kEmission]);
}

//______________________________________________________________________________
void TGLEditor::CreateSliders()
{
   fRedSlider = new TGHSlider(this, 100, kSlider1 | kScaleBoth, kHSr);
   fTrash.Add(fRedSlider);
   fRedSlider->Connect("PositionChanged(Int_t)", "TGLEditor", this, "DoSlider(Int_t)");
   fRedSlider->SetRange(0, 100);
   fRedSlider->SetPosition(Int_t(fRGBA[0] * 100));

   fGreenSlider = new TGHSlider(this, 100, kSlider1 | kScaleBoth, kHSg);
   fTrash.Add(fGreenSlider);
   fGreenSlider->Connect("PositionChanged(Int_t)", "TGLEditor", this, "DoSlider(Int_t)");
   fGreenSlider->SetRange(0, 100);
   fGreenSlider->SetPosition(Int_t(fRGBA[1] * 100));

   fBlueSlider = new TGHSlider(this, 100, kSlider1 | kScaleBoth, kHSb);
   fTrash.Add(fBlueSlider);
   fBlueSlider->Connect("PositionChanged(Int_t)", "TGLEditor", this, "DoSlider(Int_t)");
   fBlueSlider->SetRange(0, 100);
   fBlueSlider->SetPosition(Int_t(fRGBA[2] * 100));

   fAlphaSlider = new TGHSlider(this, 100, kSlider1 | kScaleBoth, kHSa);
   fTrash.Add(fAlphaSlider);
   fAlphaSlider->Connect("PositionChanged(Int_t)", "TGLEditor", this, "DoSlider(Int_t)");
   fAlphaSlider->SetRange(0, 100);
   fAlphaSlider->SetPosition(Int_t(fRGBA[3] * 100));

   fShineSlider = new TGHSlider(this, 100, kSlider1 | kScaleBoth, kHSs);
   fTrash.Add(fShineSlider);
   fShineSlider->Connect("PositionChanged(Int_t)", "TGLEditor", this, "DoSlider(Int_t)");
   fShineSlider->SetRange(0, 128);

   TGLabel *labelInfo[6] = {0};
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
void TGLEditor::SetSlidersPos()
{
   fRedSlider->SetPosition(Int_t(fRGBA[fLMode * 4] * 100));
   fGreenSlider->SetPosition(Int_t(fRGBA[fLMode * 4 + 1] * 100));
   fBlueSlider->SetPosition(Int_t(fRGBA[fLMode * 4 + 2] * 100));
   fAlphaSlider->SetPosition(Int_t(fRGBA[fLMode * 4 + 3] * 100));
   if (fRGBA[16] >= 0.f)
      fShineSlider->SetPosition(Int_t(fRGBA[16]));
}

//______________________________________________________________________________
Bool_t TGLEditor::HandleContainerNotify(Event_t *event)
{
   gVirtualX->ResizeWindow(fGLWin, event->fWidth, event->fHeight);
   DrawSphere();
   return kTRUE;
}

//______________________________________________________________________________
Bool_t TGLEditor::HandleContainerExpose(Event_t * /*event*/)
{
   DrawSphere();
   return kTRUE;
}

//______________________________________________________________________________
void TGLEditor::DrawSphere()const
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
void TGLEditor::MakeCurrent()const
{
   gVirtualGL->MakeCurrent(fGLWin, fCtx);
}

//______________________________________________________________________________
void TGLEditor::SwapBuffers()const
{
   gVirtualGL->SwapBuffers(fGLWin);
}
