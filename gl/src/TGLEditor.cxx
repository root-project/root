// @(#)root/gl:$Name:  $:$Id: TGLEditor.cxx,v 1.1 2004/09/13 09:56:33 brun Exp $
// Author:  Timur Pocheptsov  13/09/2004
   
#include <TVirtualGL.h>
#include <TVirtualX.h>
#include <TGCanvas.h>
#include <TGLayout.h>
#include <TGButton.h>
#include <TGSlider.h>
#include <TGLabel.h>


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

TGLMatView::TGLMatView(const TGWindow *parent, Window_t wid, TGLEditor *owner)
               :TGCompositeFrame(gClient, wid, parent), fOwner(owner)
{
}

Bool_t TGLMatView::HandleConfigureNotify(Event_t *event)
{
   return fOwner->HandleContainerNotify(event);
}

Bool_t TGLMatView::HandleExpose(Event_t *event)
{
   return fOwner->HandleContainerExpose(event);
}

enum EGLEditorIdent{
   kHSr,
   kHSg,
   kHSb,
   kHSa,
   kTBa
};

TGLEditor::TGLEditor(const TGWindow *parent, Int_t r, Int_t g, Int_t b, Int_t a)
               :TGCompositeFrame(parent, 100, 100, kVerticalFrame | kRaisedFrame),
                fRedSlider(0), fGreenSlider(0), fBlueSlider(0), fAlphaSlider(0),
                fApplyButton(0), fLayout(0), fLabelLayout(0),
                fIsActive(kFALSE), fRGBA(), fInfo()
{
   fRGBA[0] = r;
   fRGBA[1] = g;
   fRGBA[2] = b;
   fRGBA[3] = a;
   //////////////////////////////////////
   fViewCanvas = new TGCanvas(this, 120, 120, kSunkenFrame | kDoubleBorder);
   Window_t wid = fViewCanvas->GetViewPort()->GetId();
   fGLWin = gVirtualGL->CreateGLWindow(wid);
   fMatView = new TGLMatView(fViewCanvas->GetViewPort(), fGLWin, this);
   fCtx = gVirtualGL->CreateContext(fGLWin);
   fViewCanvas->SetContainer(fMatView);
   fViewLayout = new TGLayoutHints(kLHintsTop | kLHintsCenterX, 2, 2, 2, 2);
   AddFrame(fViewCanvas, fViewLayout);

   //sliders creation   
   fRedSlider = new TGHSlider(this, 100, kSlider1 | kScaleBoth, kHSr);
   fRedSlider->Connect("PositionChanged(Int_t)", "TGLEditor", this, "DoSlider(Int_t)");
   fRedSlider->SetRange(0, 100);
   fRedSlider->SetPosition(fRGBA[0]);

   fGreenSlider = new TGHSlider(this, 100, kSlider1 | kScaleBoth, kHSg);
   fGreenSlider->Connect("PositionChanged(Int_t)", "TGLEditor", this, "DoSlider(Int_t)");
   fGreenSlider->SetRange(0, 100);
   fGreenSlider->SetPosition(fRGBA[1]);

   fBlueSlider = new TGHSlider(this, 100, kSlider1 | kScaleBoth, kHSb);
   fBlueSlider->Connect("PositionChanged(Int_t)", "TGLEditor", this, "DoSlider(Int_t)");
   fBlueSlider->SetRange(0, 100);
   fBlueSlider->SetPosition(fRGBA[2]);

   fAlphaSlider = new TGHSlider(this, 100, kSlider1 | kScaleBoth, kHSa);
   fAlphaSlider->Connect("PositionChanged(Int_t)", "TGLEditor", this, "DoSlider(Int_t)");
   fAlphaSlider->SetRange(0, 100);
   fAlphaSlider->SetPosition(fRGBA[3]);

   fInfo[0] = new TGLabel(this, "Red :");
   fInfo[1] = new TGLabel(this, "Green :");
   fInfo[2] = new TGLabel(this, "Blue :");
   fInfo[3] = new TGLabel(this, "Alpha :");
   fLabelLayout = new TGLayoutHints(kLHintsTop | kLHintsLeft, 5, 2, 2, 5);

   fLayout = new TGLayoutHints(kLHintsTop | kLHintsLeft | kLHintsExpandX, 5, 2, 5, 10);
   AddFrame(fInfo[0], fLabelLayout);
   AddFrame(fRedSlider, fLayout);
   AddFrame(fInfo[1], fLabelLayout);
   AddFrame(fGreenSlider, fLayout);
   AddFrame(fInfo[2], fLabelLayout);
   AddFrame(fBlueSlider, fLayout);
   AddFrame(fInfo[3], fLabelLayout);
   AddFrame(fAlphaSlider, fLayout);

   //apply button creation
   fApplyButton = new TGTextButton(this, "Apply changes", kTBa);
   AddFrame(fApplyButton, fLayout);
   fApplyButton->SetState(kButtonDisabled);

   MakeCurrent();
   gVirtualGL->NewPRGL();
   gVirtualGL->FrustumGL(-0.5, 0.5, -0.5, 0.5, 1., 10.);
   gVirtualGL->LightModel(kLIGHT_MODEL_TWO_SIDE, kFALSE);
   gVirtualGL->EnableGL(kLIGHTING);
   gVirtualGL->EnableGL(kLIGHT0);
   gVirtualGL->EnableGL(kDEPTH_TEST);
   gVirtualGL->EnableGL(kCULL_FACE);
   gVirtualGL->CullFaceGL(kBACK);
   DrawSphere();
} 

TGLEditor::~TGLEditor()
{
   gVirtualGL->DeleteContext(fCtx);
   delete fRedSlider;
   delete fGreenSlider;
   delete fBlueSlider;
   delete fAlphaSlider;
   delete fLayout;
   delete fLabelLayout;
   delete fViewCanvas;
   delete fViewLayout;
   delete fMatView;
}

void TGLEditor::SetRGBA(Color_t r, Color_t g, Color_t b, Color_t a)
{
   fIsActive = kTRUE;
   fRGBA[0] = r;
   fRGBA[1] = g;
   fRGBA[2] = b;
   fRGBA[3] = a;
   fRedSlider->SetPosition(fRGBA[0]);
   fGreenSlider->SetPosition(fRGBA[1]);
   fBlueSlider->SetPosition(fRGBA[2]);
   fAlphaSlider->SetPosition(fRGBA[3]);
   DrawSphere();
}

void TGLEditor::GetRGBA(Color_t &r, Color_t &g, Color_t &b, Color_t &a)const
{
   r = fRGBA[0];
   g = fRGBA[1];
   b = fRGBA[2];
   a = fRGBA[3];
}

void TGLEditor::DoSlider(Int_t val)
{
   TGSlider *frm = (TGSlider *)gTQSender;

   if(fApplyButton->GetState() == kButtonDisabled && fIsActive)
      fApplyButton->SetState(kButtonUp);

   if (frm) {
      switch (frm->WidgetId()) {
      case kHSr:
         fRGBA[0] = val;
         break;
      case kHSg:
         fRGBA[1] = val;
         break;
      case kHSb:
         fRGBA[2] = val;
         break;
      default:
         fRGBA[3] = val;
         break;
      }
      DrawSphere();
   }
}

Bool_t TGLEditor::HandleContainerNotify(Event_t *event)
{
   gVirtualX->ResizeWindow(fGLWin, event->fWidth, event->fHeight);
   DrawSphere();
   return kTRUE;
}

Bool_t TGLEditor::HandleContainerExpose(Event_t * /*event*/)
{
   DrawSphere();
   return kTRUE;
}

void TGLEditor::DrawSphere()const
{
   MakeCurrent();
   gVirtualGL->ClearGL(0);
   gVirtualGL->ViewportGL(0, 0, fMatView->GetWidth(), fMatView->GetHeight());
   gVirtualGL->NewMVGL();
   gVirtualGL->PushGLMatrix();
   gVirtualGL->TranslateGL(0., 0., -3.);
   gVirtualGL->DrawSphere((Color_t *)fRGBA);
   gVirtualGL->PopGLMatrix();
   Float_t ligPos[] = {0.f, 0.f, 0.f, 1.f};
   gVirtualGL->GLLight(kLIGHT0, kPOSITION, ligPos);
   SwapBuffers();
}

void TGLEditor::MakeCurrent()const
{
   gVirtualGL->MakeCurrent(fGLWin, fCtx);
}

void TGLEditor::SwapBuffers()const
{
   gVirtualGL->SwapBuffers(fGLWin);
}
