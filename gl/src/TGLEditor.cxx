// @(#)root/gl:$Name:  $:$Id: TGLEditor.cxx,v 1.20 2005/11/08 19:18:18 brun Exp $
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

ClassImp(TGLColorEditor)
ClassImp(TGLGeometryEditor)
ClassImp(TGLClipEditor)
ClassImp(TGLLightEditor)
ClassImp(TGLGuideEditor)

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

//______________________________________________________________________________
TGLColorEditor::TGLColorEditor(const TGWindow *parent, TGLSAViewer *v)
               :TGCompositeFrame(parent, 100, 100, kVerticalFrame),// | kRaisedFrame),
                fViewer(v), fRedSlider(0), fGreenSlider(0), fBlueSlider(0), 
                fAlphaSlider(0), fApplyButton(0), fIsActive(kFALSE), 
                fIsLight(kFALSE), fRGBA()
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
   fApplyButton->Connect("Pressed()", "TGLColorEditor", this, "DoButton()");
   
   fApplyFamily = new TGTextButton(this, "Apply to family", kTBaf);
   fTrash.Add(fApplyFamily);
   AddFrame(fApplyFamily, widLayout);
   fApplyFamily->SetState(kButtonDisabled);
   fApplyFamily->Connect("Pressed()", "TGLColorEditor", this, "DoButton()");

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
   fApplyFamily->SetState(kButtonDisabled);

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
   fApplyButton->SetState(kButtonDisabled);
   fApplyButton->SetState(kButtonDisabled);
   fIsActive = kFALSE;
   fIsLight = kFALSE;
}

//______________________________________________________________________________
void TGLColorEditor::CreateRadioButtons()
{
   TGGroupFrame *partFrame = new TGGroupFrame(this, "Color components:", kLHintsTop | kLHintsCenterX);
   fTrash.Add(partFrame);
   partFrame->SetTitlePos(TGGroupFrame::kLeft);
   AddFrame(partFrame, fFrameLayout);
   TGMatrixLayout *ml = new TGMatrixLayout(partFrame, 0, 1, 10);
   fTrash.Add(ml);
   partFrame->SetLayoutManager(ml);

   fLightTypes[kDiffuse] = new TGRadioButton(partFrame, "Diffuse", kCPd);
   fLightTypes[kDiffuse]->Connect("Pressed()", "TGLColorEditor", this, "DoButton()");
   fLightTypes[kDiffuse]->SetToolTipText("Diffuse component of color");	
   fTrash.Add(fLightTypes[kDiffuse]);
   fLightTypes[kAmbient] = new TGRadioButton(partFrame, "Ambient", kCPa);
   fLightTypes[kAmbient]->Connect("Pressed()", "TGLColorEditor", this, "DoButton()");
   fLightTypes[kAmbient]->SetToolTipText("Ambient component of color");	
   fTrash.Add(fLightTypes[kAmbient]);
   fLightTypes[kSpecular] = new TGRadioButton(partFrame, "Specular", kCPs);
   fLightTypes[kSpecular]->Connect("Pressed()", "TGLColorEditor", this, "DoButton()");
   fLightTypes[kSpecular]->SetToolTipText("Specular component of color");	
   fTrash.Add(fLightTypes[kSpecular]);
   fLightTypes[kEmission] = new TGRadioButton(partFrame, "Emissive", kCPe);
   fLightTypes[kEmission]->Connect("Pressed()", "TGLColorEditor", this, "DoButton()");
   fLightTypes[kEmission]->SetToolTipText("Emissive component of color");	
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
TGLGeometryEditor::TGLGeometryEditor(const TGWindow *parent, TGLSAViewer *v)
                     :TGCompositeFrame(parent, 100, 100, kVerticalFrame),// | kRaisedFrame),
                      fViewer(v)
{
   fTrash.SetOwner(kTRUE);
   fIsActive = kFALSE;
   fL1 = new TGLayoutHints(kLHintsTop | kLHintsCenterX | kLHintsExpandX, 3, 3, 3, 3);
   fTrash.AddLast(fL1);
   fL2 = new TGLayoutHints(kLHintsTop | kLHintsLeft, 3, 3, 3, 3);
   fTrash.AddLast(fL2);
   CreateCenterControls();
   CreateStretchControls();
   //create button
   fApplyButton = new TGTextButton(this, "Modify object", kTBa1);
   fTrash.AddLast(fApplyButton);
   AddFrame(fApplyButton, fL1);
   fApplyButton->SetState(kButtonDisabled);
   fApplyButton->Connect("Pressed()", "TGLGeometryEditor", this, "DoButton()");
}

//______________________________________________________________________________
void TGLGeometryEditor::SetCenter(const Double_t *c)
{
   fIsActive = kTRUE;
   fApplyButton->SetState(kButtonDisabled);
   fGeomData[kCenterX]->SetNumber(c[0]);
   fGeomData[kCenterY]->SetNumber(c[1]);
   fGeomData[kCenterZ]->SetNumber(c[2]);
}

//______________________________________________________________________________
void TGLGeometryEditor::SetScale(const Double_t *s)
{
   fIsActive = kTRUE;
   fGeomData[kScaleX]->SetNumber(s[0]);
   fGeomData[kScaleY]->SetNumber(s[1]);
   fGeomData[kScaleZ]->SetNumber(s[2]);
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
   if (!fIsActive)return;
   fApplyButton->SetState(kButtonUp);
}

//______________________________________________________________________________
void TGLGeometryEditor::CreateCenterControls()
{
   TGLabel *label = new TGLabel(this, "Object's center, X:");
   fTrash.AddLast(label);
   AddFrame(label, fL2);  
   fGeomData[kCenterX] = new TGNumberEntry(this, 0.0, 8, kNExc);
   fTrash.AddLast(fGeomData[kCenterX]);
   AddFrame(fGeomData[kCenterX], fL1);
   fGeomData[kCenterX]->Connect("ValueSet(Long_t)", "TGLGeometryEditor", 
                                this, "ValueSet(Long_t)");

   label = new TGLabel(this, "Object's center, Y:");
   fTrash.AddLast(label);
   AddFrame(label, fL2);  
   fGeomData[kCenterY] = new TGNumberEntry(this, 0.0, 8, kNEyc);
   fTrash.AddLast(fGeomData[kCenterY]);
   AddFrame(fGeomData[kCenterY], fL1);
   fGeomData[kCenterY]->Connect("ValueSet(Long_t)", "TGLGeometryEditor", 
                                this, "ValueSet(Long_t)");

   label = new TGLabel(this, "Object's center, Z:");
   fTrash.AddLast(label);
   AddFrame(label, fL2);  
   fGeomData[kCenterZ] = new TGNumberEntry(this, 0.0, 8, kNEzc);
   fTrash.AddLast(fGeomData[kCenterZ]);
   AddFrame(fGeomData[kCenterZ], fL1);
   fGeomData[kCenterZ]->Connect("ValueSet(Long_t)", "TGLGeometryEditor", 
                                this, "ValueSet(Long_t)");
}

//______________________________________________________________________________
void TGLGeometryEditor::CreateStretchControls()
{
   TGLabel *label = new TGLabel(this, "Object's scale, X:");
   fTrash.AddLast(label);
   AddFrame(label, fL2);  
   fGeomData[kScaleX] = new TGNumberEntry(this, 1.0, 8, kNExs);
   fTrash.AddLast(fGeomData[kScaleX]);
   AddFrame(fGeomData[kScaleX], fL1);
   fGeomData[kScaleX]->Connect("ValueSet(Long_t)", "TGLGeometryEditor", 
                               this, "ValueSet(Long_t)");

   label = new TGLabel(this, "Object's scale, Y:");
   fTrash.AddLast(label);
   AddFrame(label, fL2);  
   fGeomData[kScaleY] = new TGNumberEntry(this, 1.0, 8, kNEys);
   fTrash.AddLast(fGeomData[kScaleY]);
   AddFrame(fGeomData[kScaleY], fL1);
   fGeomData[kScaleY]->Connect("ValueSet(Long_t)", "TGLGeometryEditor", 
                               this, "ValueSet(Long_t)");

   label = new TGLabel(this, "Object's scale, Z:");
   fTrash.AddLast(label);
   AddFrame(label, fL2);  
   fGeomData[kScaleZ] = new TGNumberEntry(this, 1.0, 8, kNEzs);
   fTrash.AddLast(fGeomData[kScaleZ]);
   AddFrame(fGeomData[kScaleZ], fL1);
   fGeomData[kScaleZ]->Connect("ValueSet(Long_t)", "TGLGeometryEditor", 
                               this, "ValueSet(Long_t)");

   fGeomData[kScaleX]->SetLimits(TGNumberFormat::kNELLimitMin, 0.1);
   fGeomData[kScaleY]->SetLimits(TGNumberFormat::kNELLimitMin, 0.1);
   fGeomData[kScaleZ]->SetLimits(TGNumberFormat::kNELLimitMin, 0.1);
}

//______________________________________________________________________________
TGLClipEditor::TGLClipEditor(const TGWindow *parent, TGLSAViewer *v) : 
   TGCompositeFrame(parent, 100, 100, kVerticalFrame),
                    fViewer(v), fCurrentClip(kClipNone)
{
   fTrash.SetOwner(kTRUE);
   fL1 = new TGLayoutHints(kLHintsTop | kLHintsCenterX | kLHintsExpandX, 3, 3, 3, 3);
   fTrash.AddLast(fL1);
   fL2 = new TGLayoutHints(kLHintsTop | kLHintsLeft, 3, 3, 3, 3);
   fTrash.AddLast(fL2);
   CreateControls();
}

//______________________________________________________________________________
void TGLClipEditor::CreateControls()
{
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
	HideFrame(fPlanePropFrame);
	HideFrame(fBoxPropFrame);
}

//______________________________________________________________________________
void TGLClipEditor::ClipValueChanged(Long_t)
{
   fApplyButton->SetState(kButtonUp);
}

//______________________________________________________________________________
void TGLClipEditor::ClipTypeChanged(Int_t id)
{  
   switch(id) { // Radio button ids run from 1
      case(1): {
         SetCurrent(kClipNone);
         fEdit->SetState(kButtonDisabled);
         break;
      }
      case(2): {
         SetCurrent(kClipPlane);
         fEdit->SetState(kButtonUp);
         break;
      }
      case(3): {
         SetCurrent(kClipBox);
         fEdit->SetState(kButtonUp);
         break;
      }
   }

   // Internal GUI change - need to update the viewer
   UpdateViewer();
}

//______________________________________________________________________________
void TGLClipEditor::UpdateViewer()
{
   // Generic 'something changed' event - everything is pulled from GUI
   fViewer->ProcessGUIEvent(kTBcpm);
   fApplyButton->SetState(kButtonDisabled);
}

//______________________________________________________________________________
void TGLClipEditor::GetState(EClipType type, std::vector<Double_t> & data) const
{
   UInt_t i;
   data.clear();
   if (type == kNone) {
      // Nothing to do
   } else if (type == kClipPlane) {
      for (i=0; i<4; i++) {
         data.push_back(fPlaneProp[i]->GetNumber());
      }
   } else if (type == kClipBox) {
      for (i=0; i<6; i++) {
         data.push_back(fBoxProp[i]->GetNumber());
      }
   } else {
      Error("TGLClipEditor::GetClipState", "Invalid clip type");
   }
}

//______________________________________________________________________________
void TGLClipEditor::SetState(EClipType type, const std::vector<Double_t> & data)
{
   UInt_t i;
   if (type == kNone) {
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
   type = fCurrentClip;
   edit = fEdit->IsDown();
}

//______________________________________________________________________________
void TGLClipEditor::SetCurrent(EClipType type)
{
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
}

//______________________________________________________________________________
TGLLightEditor::TGLLightEditor(const TGWindow *parent, TGLSAViewer *v)
               :TGCompositeFrame(parent, 100, 100, kVerticalFrame),// | kRaisedFrame),
                fViewer(v)
{
   fTrash.SetOwner(kTRUE);
   TGGroupFrame *ligFrame = new TGGroupFrame(this, "Sources", kLHintsTop | kLHintsCenterX);
   fTrash.Add(ligFrame);
   ligFrame->SetTitlePos(TGGroupFrame::kLeft);
   TGLayoutHints *l = new TGLayoutHints(kLHintsTop | kLHintsCenterX | kLHintsExpandX, 3, 3, 3, 3);
   AddFrame(ligFrame, l);
   
   TGMatrixLayout *ml = new TGMatrixLayout(ligFrame, 0, 1, 10);
   fTrash.Add(ml);
   ligFrame->SetLayoutManager(ml);

   fLights[kTop] = new TGCheckButton(ligFrame, "Top", kTBTop);
   fLights[kTop]->Connect("Clicked()", "TGLLightEditor", this, "DoButton()");
   fLights[kTop]->SetState(kButtonDown);
   fTrash.Add(fLights[kTop]);
   fLights[kRight] = new TGCheckButton(ligFrame, "Right", kTBRight);
   fLights[kRight]->Connect("Clicked()", "TGLLightEditor", this, "DoButton()");
   fLights[kRight]->SetState(kButtonDown);
   fTrash.Add(fLights[kRight]);
   fLights[kBottom] = new TGCheckButton(ligFrame, "Bottom", kTBBottom);
   fLights[kBottom]->Connect("Clicked()", "TGLLightEditor", this, "DoButton()");
   fLights[kBottom]->SetState(kButtonDown);
   fTrash.Add(fLights[kBottom]);
   fLights[kLeft] = new TGCheckButton(ligFrame, "Left", kTBLeft);
   fLights[kLeft]->Connect("Clicked()", "TGLLightEditor", this, "DoButton()");
   fLights[kLeft]->SetState(kButtonDown);
   fTrash.Add(fLights[kLeft]);
   fLights[kFront] = new TGCheckButton(ligFrame, "Front", kTBFront);
   fLights[kFront]->Connect("Clicked()", "TGLLightEditor", this, "DoButton()");
   fLights[kFront]->SetState(kButtonDown);
   fTrash.Add(fLights[kFront]);

   ligFrame->AddFrame(fLights[kTop]);
   ligFrame->AddFrame(fLights[kRight]);
   ligFrame->AddFrame(fLights[kBottom]);
   ligFrame->AddFrame(fLights[kLeft]);
   ligFrame->AddFrame(fLights[kFront]);
}

//______________________________________________________________________________
void TGLLightEditor::DoButton()
{
   TGButton *btn = (TGButton *) gTQSender;
   Int_t id = btn->WidgetId();
   fViewer->ProcessGUIEvent(id);
}

//______________________________________________________________________________
TGLGuideEditor::TGLGuideEditor(const TGWindow *parent, TGLSAViewer *v) :
   TGCompositeFrame(parent, 100, 100, kVerticalFrame),
   fViewer(v),  fAxesContainer(0), fReferenceContainer(0),
   fReferenceOn(0),
   fL1(0), fL2(0)
{
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
void TGLGuideEditor::Update()
{
   fViewer->ProcessGUIEvent(kTBGuide);
   UpdateReferencePos();
}

//______________________________________________________________________________
void TGLGuideEditor::GetState(EAxesType & axesType, Bool_t & referenceOn, TGLVertex3 & referencePos) const
{
   // Button ids run from 1
   for (Int_t i = 1; i < 4; i++) { 
      TGButton * button = fAxesContainer->GetButton(i);
      if (button && button->IsDown()) {
         axesType = EAxesType(i-1);
      }
   }
   referenceOn = fReferenceOn->IsDown();
   referencePos.Set(fReferencePos[0]->GetNumber(),
                    fReferencePos[1]->GetNumber(),
                    fReferencePos[2]->GetNumber());
}

//______________________________________________________________________________
void TGLGuideEditor::SetState(EAxesType axesType, Bool_t referenceOn, const TGLVertex3 & referencePos)
{
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
   fReferencePos[0]->SetState(fReferenceOn->IsDown());
   fReferencePos[1]->SetState(fReferenceOn->IsDown());
   fReferencePos[2]->SetState(fReferenceOn->IsDown());
}
