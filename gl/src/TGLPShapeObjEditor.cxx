// @(#)root/gl:$Name:  $:$Id: TGLPShapeObjEditor.cxx,v 1.2 2006/09/26 13:44:56 rdm Exp $
// Author: Matevz Tadel   25/09/2006

#include <cstring>

#include "TGLPShapeObjEditor.h"
#include "TGedEditor.h"

#include "TG3DLine.h"
#include "TGButton.h"
#include "TGButtonGroup.h"
#include "TString.h"
#include "TGLabel.h"
#include "TClass.h"
#include "TGCanvas.h"
#include "TGTab.h"
#include "TGSlider.h"
#include "TGNumberEntry.h"
#include "TGButtonGroup.h"

#include "TVirtualGL.h"
#include "TVirtualX.h"
#include "TGLViewer.h"
#include "TGLUtil.h"
#include "TGLPhysicalShape.h"


class TGLMatView : public TGCompositeFrame {
// Helper class to handle GL window with colored sphere.
private:
   TGLPShapeObjEditor *fOwner;
public:
   TGLMatView(const TGWindow *parent, Window_t wid, TGLPShapeObjEditor *owner);
   Bool_t HandleConfigureNotify(Event_t *event);
   Bool_t HandleExpose(Event_t *event);

private:
   TGLMatView(const TGLMatView &);
   TGLMatView & operator = (const TGLMatView &);
};

//______________________________________________________________________________
TGLMatView::TGLMatView(const TGWindow *parent, Window_t wid, TGLPShapeObjEditor *owner)
               :TGCompositeFrame(gClient, wid, parent), fOwner(owner)
{
   // Constructor.
   AddInput(kExposureMask | kStructureNotifyMask);
}

//______________________________________________________________________________
Bool_t TGLMatView::HandleConfigureNotify(Event_t *event)
{
   // Pass to fOwner.
   return fOwner->HandleContainerNotify(event);
}


//______________________________________________________________________________
Bool_t TGLMatView::HandleExpose(Event_t *event)
{
   // Pass to fOwner.
   return fOwner->HandleContainerExpose(event);
}

//______________________________________________________________________________
ClassImp(TGLPShapeObjEditor)

enum EGeometry {
      kCenterX,
      kCenterY,
      kCenterZ,
      kScaleX,
      kScaleY,
      kScaleZ,
      kTot
};


enum EApplyButtonIds {
      kTBcp,
      kTBcpm,
      kTBda,
      kTBa,
      kTBaf,
      kTBTop,
      kTBRight,
      kTBBottom,
      kTBLeft,
      kTBFront,
      kTBa1,
      kTBGuide
};

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


//A lot of raw pointers/naked new-expressions - good way to discredit C++ (or C++ programmer :) ) :(
//ROOT has system to cleanup - I'll try  to use it
//______________________________________________________________________________
TGLPShapeObjEditor::TGLPShapeObjEditor(const TGWindow *p,  Int_t width, Int_t height, UInt_t options, Pixel_t back)
   : TGedFrame(p,  width, height, options | kVerticalFrame, back),
     fLb(kLHintsTop | kLHintsCenterX | kLHintsExpandX, 2, 2, 3, 3), //button
     fLe(kLHintsTop | kLHintsCenterX | kLHintsExpandX, 0, 0, 3, 3), //entries
     fLl(kLHintsLeft, 0, 8, 6, 0), // labels
     fLs(kLHintsTop | kLHintsCenterX, 2, 2, 0, 0),  ///sliders
     fIsActive(kTRUE), fGeoFrame(0),fGeoApplyButton(0),
     fColorFrame(0),
     fRedSlider(0), fGreenSlider(0), fBlueSlider(0), fAlphaSlider(0), fShineSlider(0),
     fColorApplyButton(0), fColorApplyFamily(0),
     fIsLight(kFALSE), fRGBA(),
     fGLWin(0),
     fPShapeObj(0)
{
   // Constructor of TGLPhysicalShape editor GUI.

   fRGBA[12] = 0.f, fRGBA[13] = 0.f, fRGBA[14] = 0.f;
   fRGBA[15] = 1.f, fRGBA[16] = 60.f;

   CreateColorControls();
   CreateGeoControls();
}

//______________________________________________________________________________
TGLPShapeObjEditor::~TGLPShapeObjEditor()
{
   // Destroy color editor GUI component.
   delete fMatView;
   gVirtualGL->DeleteContext(fCtx);
}

//______________________________________________________________________________
void TGLPShapeObjEditor::SetModel(TObject* obj)
{
   // Sets model or disables/hides viewer.

   fPShapeObj = 0;

   fPShapeObj = static_cast<TGLPShapeObj *>(obj);

   SetRGBA(fPShapeObj->fPShape->Color());
   SetCenter(fPShapeObj->fPShape->GetTranslation().CArr());
   SetScale(fPShapeObj->fPShape->GetScale().CArr());
}

//______________________________________________________________________________
void TGLPShapeObjEditor::SetCenter(const Double_t *c)
{
   // Set internal center data from 3 component 'c'.

   fGeoApplyButton->SetState(kButtonDisabled);
   fGeomData[kCenterX]->SetNumber(c[0]);
   fGeomData[kCenterY]->SetNumber(c[1]);
   fGeomData[kCenterZ]->SetNumber(c[2]);
}

//______________________________________________________________________________
void TGLPShapeObjEditor::SetScale(const Double_t *s)
{
   // Set internal scale data from 3 component 'c'.

   fGeomData[kScaleX]->SetNumber(s[0]);
   fGeomData[kScaleY]->SetNumber(s[1]);
   fGeomData[kScaleZ]->SetNumber(s[2]);
}

//______________________________________________________________________________
void TGLPShapeObjEditor::GeoDisable()
{
   // Disable "Apply" button.

   fIsActive = kFALSE;
   fGeoApplyButton->SetState(kButtonDisabled);
}

//______________________________________________________________________________
void TGLPShapeObjEditor::DoGeoButton()
{
   // Process 'Apply' - update the viewer object from GUI.

   if (TGButton *btn = (TGButton *)gTQSender) {
      Int_t wid = btn->WidgetId();
      //      fViewer->ProcessGUIEvent(wid);

      TGLVertex3 trans;
      TGLVector3 scale;
      GetObjectData(trans.Arr(), scale.Arr());
      fPShapeObj->fViewer->SetSelectedGeom(trans,scale);
      if (wid == kTBa1) {
         fGeoApplyButton->SetState(kButtonDisabled);
      }
   }
}

//______________________________________________________________________________
void TGLPShapeObjEditor::GetObjectData(Double_t *center, Double_t *scale)
{
   // Extract the GUI object data, return center in 3 component 'center'
   // scale in 3 component 'scale'.

   center[0] = fGeomData[kCenterX]->GetNumber();
   center[1] = fGeomData[kCenterY]->GetNumber();
   center[2] = fGeomData[kCenterZ]->GetNumber();
   scale[0] = fGeomData[kScaleX]->GetNumber();
   scale[1] = fGeomData[kScaleY]->GetNumber();
   scale[2] = fGeomData[kScaleZ]->GetNumber();
}

//______________________________________________________________________________
void TGLPShapeObjEditor::GeoValueSet(Long_t)
{
   // Process setting of value in edit box - activate 'Apply' button.

   if (!fIsActive)return;
   fGeoApplyButton->SetState(kButtonUp);
}

//______________________________________________________________________________
void TGLPShapeObjEditor::CreateGeoControls()
{
   // Create GUI for setting scale and position. 

   fGeoFrame = CreateEditorTabSubFrame("Geometry");

   TGLabel *label=0;
   // postion containers
   TGGroupFrame* container = new TGGroupFrame(fGeoFrame, "Object position:");
   container->SetTitlePos(TGGroupFrame::kLeft);
   fGeoFrame->AddFrame(container, new TGLayoutHints(kLHintsTop | kLHintsCenterX | kLHintsExpandX, 8, 8, 3, 3));//-
   TGLayoutHints lh =  TGLayoutHints(kLHintsTop | kLHintsCenterX | kLHintsExpandX, 0, 0, 0, 0);

   TGHorizontalFrame* hf;

   hf = new TGHorizontalFrame(container);
   label = new TGLabel(hf, "X:");
   hf->AddFrame(label, new TGLayoutHints(fLl));
   fGeomData[kCenterX] = new TGNumberEntry(hf, 0.0, 8, kNExc);
   hf->AddFrame(fGeomData[kCenterX], new TGLayoutHints(fLe));
   fGeomData[kCenterX]->Connect("ValueSet(Long_t)", "TGLPShapeObjEditor",
                                this, "GeoValueSet(Long_t)");
   container->AddFrame(hf, new TGLayoutHints(lh));

   hf = new TGHorizontalFrame(container);
   label = new TGLabel(hf, "Y:");
   hf->AddFrame(label, new TGLayoutHints(fLl));
   fGeomData[kCenterY] = new TGNumberEntry(hf, 0.0, 8, kNEyc);
   hf->AddFrame(fGeomData[kCenterY], new TGLayoutHints(fLe));
   fGeomData[kCenterY]->Connect("ValueSet(Long_t)", "TGLPShapeObjEditor",
                                this, "GeoValueSet(Long_t)");
   container->AddFrame(hf, new TGLayoutHints(lh));

   hf = new TGHorizontalFrame(container);
   hf->AddFrame(new TGLabel(hf, "Z:"), new TGLayoutHints(fLl));
   fGeomData[kCenterZ] = new TGNumberEntry(hf, 1.0, 8, kNEzc);
   hf->AddFrame(fGeomData[kCenterZ], new TGLayoutHints(fLe));
   fGeomData[kCenterZ]->Connect("ValueSet(Long_t)", "TGLPShapeObjEditor",
                                this, "GeoValueSet(Long_t)");
   container->AddFrame(hf, new TGLayoutHints(lh));

   // Create object scale GUI
   TGGroupFrame* osf = new TGGroupFrame(fGeoFrame, "Object scale:", kLHintsTop | kLHintsCenterX);
   osf->SetTitlePos(TGGroupFrame::kLeft);
   fGeoFrame->AddFrame(osf, new TGLayoutHints(kLHintsTop | kLHintsCenterX | kLHintsExpandX, 8, 8, 3, 3));

   hf = new TGHorizontalFrame(osf);
   hf->AddFrame(new TGLabel(hf, "X:"),new TGLayoutHints(fLl));
   fGeomData[kScaleX] = new TGNumberEntry(hf, 1.0, 5, kNExs);
   hf->AddFrame(fGeomData[kScaleX], new TGLayoutHints(fLe));
   fGeomData[kScaleX]->Connect("ValueSet(Long_t)", "TGLPShapeObjEditor",
                               this, "GeoValueSet(Long_t)");
   osf->AddFrame(hf, new TGLayoutHints(lh));

   hf = new TGHorizontalFrame(osf);
   hf->AddFrame(new TGLabel(hf, "Y:"),new TGLayoutHints(fLl));
   fGeomData[kScaleY] = new TGNumberEntry(hf, 1.0, 5, kNEys);
   hf->AddFrame(fGeomData[kScaleY], new TGLayoutHints(fLe));
   fGeomData[kScaleY]->Connect("ValueSet(Long_t)", "TGLPShapeObjEditor",
                               this, "GeoValueSet(Long_t)");
   osf->AddFrame(hf, new TGLayoutHints(lh));

   hf = new TGHorizontalFrame(osf);
   hf->AddFrame(new TGLabel(hf, "Z:"),new TGLayoutHints(fLl));
   fGeomData[kScaleZ] = new TGNumberEntry(hf, 1.0, 5, kNEzs);
   hf->AddFrame(fGeomData[kScaleZ], new TGLayoutHints(fLe));
   fGeomData[kScaleZ]->Connect("ValueSet(Long_t)", "TGLPShapeObjEditor",
                               this, "GeoValueSet(Long_t)");
   osf->AddFrame(hf, new TGLayoutHints(lh));

   hf = new TGHorizontalFrame(osf);
   fGeomData[kScaleX]->SetLimits(TGNumberFormat::kNELLimitMin, 0.1);
   fGeomData[kScaleY]->SetLimits(TGNumberFormat::kNELLimitMin, 0.1);
   fGeomData[kScaleZ]->SetLimits(TGNumberFormat::kNELLimitMin, 0.1);
   osf->AddFrame(hf, new TGLayoutHints(lh));

   //create button

   fGeoApplyButton = new TGTextButton(fGeoFrame, "Modify object", kTBa1);
   fGeoFrame->AddFrame(fGeoApplyButton, new TGLayoutHints(fLb));
   fGeoApplyButton->SetState(kButtonDisabled);
   fGeoApplyButton->Connect("Pressed()", "TGLPShapeObjEditor", this, "DoGeoButton()");
}

//______________________________________________________________________________
void TGLPShapeObjEditor::SetRGBA(const Float_t *rgba)
{
   // Set color sliders from 17 component 'rgba'.

   fColorApplyButton->SetState(kButtonDisabled);
   fColorApplyFamily->SetState(kButtonDisabled);

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
      //fAlphaSlider->SetPosition(Int_t(fRGBA[3] * 100));
      fShineSlider->SetPosition(Int_t(fRGBA[16]));
   }

   fRedSlider->SetPosition(Int_t(fRGBA[fLMode * 4] * 100));
   fGreenSlider->SetPosition(Int_t(fRGBA[fLMode * 4 + 1] * 100));
   fBlueSlider->SetPosition(Int_t(fRGBA[fLMode * 4 + 2] * 100));

   DrawSphere();
}

//______________________________________________________________________________
void TGLPShapeObjEditor::DoColorSlider(Int_t val)
{
   // Process slider movement.

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
            fColorApplyButton->SetState(kButtonUp);
            if (!fIsLight) fColorApplyFamily->SetState(kButtonUp);
         }
         DrawSphere();
      }
   }
}

//______________________________________________________________________________
void TGLPShapeObjEditor::DoColorButton()
{
   // Process button action.

   TGButton *btn = (TGButton *) gTQSender;
   Int_t id = btn->WidgetId();

   switch (id) {
   case kCPd:
      fLightTypes[fLMode]->SetState(kButtonUp);
      fLMode = kDiffuse;
      SetColorSlidersPos();
      break;
   case kCPa:
      fLightTypes[fLMode]->SetState(kButtonUp);
      fLMode = kAmbient;
      SetColorSlidersPos();
      break;
   case kCPs:
      fLightTypes[fLMode]->SetState(kButtonUp);
      fLMode = kSpecular;
      SetColorSlidersPos();
      break;
   case kCPe:
      fLightTypes[fLMode]->SetState(kButtonUp);
      fLMode = kEmission;
      SetColorSlidersPos();
      break;
   case kTBa:
      fColorApplyButton->SetState(kButtonDisabled);
      fColorApplyFamily->SetState(kButtonDisabled);
      fPShapeObj->fViewer->SetSelectedColor(GetRGBA());
      break;
   case kTBaf:
      fColorApplyButton->SetState(kButtonDisabled);
      fColorApplyFamily->SetState(kButtonDisabled);
      fPShapeObj->fViewer->SetColorOnSelectedFamily(GetRGBA());
      break;
   }
   DrawSphere();
}

//______________________________________________________________________________
void TGLPShapeObjEditor::CreateMaterialView()
{
   // Small gl-window with sphere.

   TGCanvas *viewCanvas = new TGCanvas(fColorFrame, 120, 120, kSunkenFrame | kDoubleBorder);
   Window_t wid = viewCanvas->GetViewPort()->GetId();
   fGLWin = gVirtualGL->CreateGLWindow(wid);

   fMatView = new TGLMatView(viewCanvas->GetViewPort(), fGLWin, this);
   fCtx = gVirtualGL->CreateContext(fGLWin);
   viewCanvas->SetContainer(fMatView);
   fColorFrame->AddFrame(viewCanvas, new TGLayoutHints(kLHintsTop | kLHintsCenterX, 2, 0, 2, 2));
}

//______________________________________________________________________________
void TGLPShapeObjEditor::CreateColorRadioButtons()
{
   // Create Diffuse/Ambient/Specular/Emissive radio buttons and sub-frames.

   TGGroupFrame *partFrame = new TGGroupFrame(fColorFrame, "Color components:", kLHintsTop | kLHintsCenterX);
   fColorFrame->AddFrame(partFrame, new TGLayoutHints(kLHintsTop | kLHintsCenterX, 2, 0, 2, 2));

   partFrame->SetTitlePos(TGGroupFrame::kLeft);
   TGMatrixLayout *ml = new TGMatrixLayout(partFrame, 0, 1, 10);
   partFrame->SetLayoutManager(ml);

   // partFrame will delete the layout manager ml for us so don't add to fTrash
   fLightTypes[kDiffuse] = new TGRadioButton(partFrame, "Diffuse", kCPd);
   fLightTypes[kDiffuse]->Connect("Pressed()", "TGLPShapeObjEditor", this, "DoColorButton()");
   fLightTypes[kDiffuse]->SetToolTipText("Diffuse component of color");
   partFrame->AddFrame(fLightTypes[kDiffuse]);

   fLightTypes[kAmbient] = new TGRadioButton(partFrame, "Ambient", kCPa);
   fLightTypes[kAmbient]->Connect("Pressed()", "TGLPShapeObjEditor", this, "DoColorButton()");
   fLightTypes[kAmbient]->SetToolTipText("Ambient component of color");
   partFrame->AddFrame(fLightTypes[kAmbient]);

   fLightTypes[kSpecular] = new TGRadioButton(partFrame, "Specular", kCPs);
   fLightTypes[kSpecular]->Connect("Pressed()", "TGLPShapeObjEditor", this, "DoColorButton()");
   fLightTypes[kSpecular]->SetToolTipText("Specular component of color");
   partFrame->AddFrame(fLightTypes[kSpecular]);

   fLightTypes[kEmission] = new TGRadioButton(partFrame, "Emissive", kCPe);
   fLightTypes[kEmission]->Connect("Pressed()", "TGLPShapeObjEditor", this, "DoColorButton()");
   fLightTypes[kEmission]->SetToolTipText("Emissive component of color");
   partFrame->AddFrame(fLightTypes[kEmission]);

   fLMode = kDiffuse;
   fLightTypes[fLMode]->SetState(kButtonDown);
}

//______________________________________________________________________________
void TGLPShapeObjEditor::CreateColorSliders()
{
   // Create GUI for setting light color.

   UInt_t sw = 120; //fColorFrame->GetDefalutWidth();,

   // Create Red/Green/BlueAlpha/Shine sliders
   fColorFrame->AddFrame(new TGLabel(fColorFrame, "Red :"), new TGLayoutHints(kLHintsTop | kLHintsLeft, 5, 0, 0, 0));
   fRedSlider = new TGHSlider(fColorFrame, sw, kSlider1 | kScaleBoth, kHSr);
   fRedSlider->Connect("PositionChanged(Int_t)", "TGLPShapeObjEditor", this, "DoColorSlider(Int_t)");
   fRedSlider->SetRange(0, 100);
   fRedSlider->SetPosition(Int_t(fRGBA[0] * 100));
   fColorFrame->AddFrame(fRedSlider, new TGLayoutHints(fLs));


   fColorFrame->AddFrame(new TGLabel(fColorFrame, "Green :"), new TGLayoutHints(kLHintsTop | kLHintsLeft, 5, 0, 0, 0));
   fGreenSlider = new TGHSlider(fColorFrame, sw, kSlider1 | kScaleBoth, kHSg);
   fGreenSlider->Connect("PositionChanged(Int_t)", "TGLPShapeObjEditor", this, "DoColorSlider(Int_t)");
   fGreenSlider->SetRange(0, 100);
   fGreenSlider->SetPosition(Int_t(fRGBA[1] * 100));
   fColorFrame->AddFrame(fGreenSlider, new TGLayoutHints(fLs));


   fColorFrame->AddFrame(new TGLabel(fColorFrame, "Blue :"), new TGLayoutHints(kLHintsTop | kLHintsLeft, 5, 0, 0, 0));
   fBlueSlider = new TGHSlider(fColorFrame, sw, kSlider1 | kScaleBoth, kHSb);
   fBlueSlider->Connect("PositionChanged(Int_t)", "TGLPShapeObjEditor", this, "DoColorSlider(Int_t)");
   fBlueSlider->SetRange(0, 100);
   fBlueSlider->SetPosition(Int_t(fRGBA[2] * 100));
   fColorFrame->AddFrame(fBlueSlider, new TGLayoutHints(fLs));

   fColorFrame->AddFrame(new TGLabel(fColorFrame, "Shine :"), new TGLayoutHints(kLHintsTop | kLHintsLeft, 5, 0, 0, 0));
   fShineSlider = new TGHSlider(fColorFrame, sw, kSlider1 | kScaleBoth, kHSs);
   fShineSlider->Connect("PositionChanged(Int_t)", "TGLPShapeObjEditor", this, "DoColorSlider(Int_t)");
   fShineSlider->SetRange(0, 128);
   fColorFrame->AddFrame(fShineSlider, new TGLayoutHints(fLs));

}

//______________________________________________________________________________
void TGLPShapeObjEditor::SetColorSlidersPos()
{
   // Update GUI sliders from internal data.

   fRedSlider->SetPosition(Int_t(fRGBA[fLMode * 4] * 100));
   fGreenSlider->SetPosition(Int_t(fRGBA[fLMode * 4 + 1] * 100));
   fBlueSlider->SetPosition(Int_t(fRGBA[fLMode * 4 + 2] * 100));
   //   fAlphaSlider->SetPosition(Int_t(fRGBA[fLMode * 4 + 3] * 100));

   if (fRGBA[16] >= 0.f)
      fShineSlider->SetPosition(Int_t(fRGBA[16]));
}

//______________________________________________________________________________
Bool_t TGLPShapeObjEditor::HandleContainerNotify(Event_t * /*event*/)
{
   // Handle resize event.

   DrawSphere();
   return kTRUE;
}

//______________________________________________________________________________
Bool_t TGLPShapeObjEditor::HandleContainerExpose(Event_t * /*event*/)
{
   // Handle expose (show) event.

   DrawSphere();
   return kTRUE;
}

//______________________________________________________________________________
void TGLPShapeObjEditor::DrawSphere()const
{
   // Draw local sphere reflecting current color options.

   gVirtualGL->MakeCurrent(fGLWin, fCtx);
   gVirtualGL->ClearGL(0);
   if (fIsActive) {
      gVirtualGL->ViewportGL(0, 0, fMatView->GetWidth(), fMatView->GetHeight());
      gVirtualGL->NewMVGL();
      Float_t ligPos[] = {0.f, 0.f, 0.f, 1.f};
      gVirtualGL->GLLight(kLIGHT0, kPOSITION, ligPos);
      gVirtualGL->TranslateGL(0., 0., -3.);
      gVirtualGL->DrawSphere(fRGBA);
   }

   gVirtualGL->SwapBuffers(fGLWin);
}

//______________________________________________________________________________
void TGLPShapeObjEditor::CreateColorControls()
{
   // Create widgets to chhos colors componnet and its RGBA values on fGedEditor
   // model or family it belongs to.

   fColorFrame = this;
   CreateMaterialView();

   CreateColorRadioButtons();

   CreateColorSliders();


   //apply button creation
   fColorApplyButton = new TGTextButton(fColorFrame, "Apply", kTBa);
   fColorFrame->AddFrame(fColorApplyButton, new TGLayoutHints(fLb));
   fColorApplyButton->SetState(kButtonDisabled);
   fColorApplyButton->Connect("Pressed()", "TGLPShapeObjEditor", this, "DoColorButton()");
   //apply to family button creation
   fColorApplyFamily = new TGTextButton(fColorFrame, "Apply to family", kTBaf);
   fColorFrame->AddFrame(fColorApplyFamily, new TGLayoutHints(fLb));
   fColorApplyFamily->SetState(kButtonDisabled);
   fColorApplyFamily->Connect("Pressed()", "TGLPShapeObjEditor", this, "DoColorButton()");

   //Init GL
   gVirtualGL->MakeCurrent(fGLWin, fCtx);

   gVirtualGL->NewPRGL();
   gVirtualGL->FrustumGL(-0.5, 0.5, -0.5, 0.5, 1., 10.);
   gVirtualGL->EnableGL(kLIGHTING);
   gVirtualGL->EnableGL(kLIGHT0);
   gVirtualGL->EnableGL(kDEPTH_TEST);
   gVirtualGL->EnableGL(kCULL_FACE);
   gVirtualGL->CullFaceGL(kBACK);

   DrawSphere();
}
