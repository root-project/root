// @(#)root/eve:$Id: triangleset.C 26568 2008-12-01 20:55:50Z matevz $
// Author: Alja Mrak-Tadel

// Calorimeter detailed view by using TEveCaloDataVec as data-source.

#if defined(__CINT__) && !defined(__MAKECINT__)

{
   gSystem->CompileMacro("cms_calo_detail.C");
   cms_calo_detail();
}

#else

#include <TEveManager.h>
#include <TEveCalo.h>
#include <TEveCaloData.h>
#include <TEveLegoOverlay.h>
#include <TEveLegoEventHandler.h>

#include <TGLViewer.h>
#include <TGLOverlayButton.h>

#include <TAxis.h>

#ifdef WIN32
#include <Windows4root.h>
#pragma comment(lib, "OpenGL32.lib")
#endif

#include <GL/gl.h>


class ButtFaker : public TGLOverlayButton
{
   ButtFaker(const ButtFaker&);            // Not implemented
   ButtFaker& operator=(const ButtFaker&); // Not implemented

public:
   Bool_t fShowLegend;

   ButtFaker(TGLViewerBase *parent) :
      TGLOverlayButton(parent, "Legend", 10, 200, 50, 16),
      fShowLegend(kTRUE)
   {}

   virtual ~ButtFaker() {}

   virtual void Clicked(TGLViewerBase *viewer)
   {
      fShowLegend = !fShowLegend;
      TGLOverlayButton::Clicked(viewer);
   }

   virtual void Render(TGLRnrCtx& rnrCtx)
   {
      TGLOverlayButton::Render(rnrCtx);

      if (fShowLegend)
      {
         // Render other stuff here, see TGLOverlayButton.
         // I guess you might want to move to pixel coordinates.

         glMatrixMode(GL_PROJECTION);
         glPushMatrix();
         glLoadIdentity();
         const TGLRect& vp = rnrCtx.RefCamera().RefViewport();
         glOrtho(vp.X(), vp.Width(), vp.Y(), vp.Height(), 0, 1);
         glMatrixMode(GL_MODELVIEW);
         glPushMatrix();
         glLoadIdentity();

         glColor4f(1, 0, 0, 1);
         fFont.PreRender(kFALSE);
         glPushMatrix();
         glTranslatef(20, vp.Height()-30, 0);
         glRasterPos2i(0, 0);
         fFont.Render("Ooogladoogla");
         glPopMatrix();
         fFont.PostRender();

         glMatrixMode(GL_PROJECTION);
         glPopMatrix();
         glMatrixMode(GL_MODELVIEW);
         glPopMatrix();
      }
   }

   ClassDef(ButtFaker,0);
};


void cms_calo_detail()
{
  TEveManager::Create();

  TGLViewer* v = gEve->GetDefaultGLViewer(); // Default
  v->SetCurrentCamera(TGLViewer::kCameraPerspXOY);
  v->SetEventHandler(new TEveLegoEventHandler("Lego", (TGWindow*)v->GetGLWidget(), (TObject*)v));

  // data

  TEveCaloDataVec* data = new TEveCaloDataVec(2);

  data->RefSliceInfo(0).Setup("ECAL", 0.3, kRed);
  data->RefSliceInfo(1).Setup("HCAL", 0.1, kYellow);

  data->AddTower(0.12, 0.14, 0.45, 0.47);
  data->FillSlice(0, 12);
  data->FillSlice(1, 3);

  data->AddTower(0.125, 0.145, 0.43, 0.45);
  data->FillSlice(0, 4);
  data->FillSlice(1, 7);

  data->AddTower(0.10, 0.12, 0.45, 0.47);
  data->FillSlice(0, 6);
  data->FillSlice(1, 0);

  data->SetAxisFromBins();
  // set eta, phi axis title with symbol.ttf font
  data->GetEtaBins()->SetTitle("X[cm]");
  data->GetEtaBins()->SetTitleSize(0.1);
  data->GetPhiBins()->SetTitle("Y[cm]");
  data->GetPhiBins()->SetTitleColor(kGreen);
  data->DataChanged();

  // add offset
  Double_t etaMin, etaMax;
  Double_t phiMin, phiMax;
  data->GetEtaLimits(etaMin, etaMax);
  data->GetPhiLimits(phiMin, phiMax);
  Float_t offe = 0.1*(etaMax -etaMin);
  Float_t offp = 0.1*(etaMax -etaMin);
  data->AddTower(etaMin -offe, etaMax +offe, phiMin -offp , phiMax +offp);


  // lego
  TEveCaloLego* lego = new TEveCaloLego(data);
  lego->SetAutoRebin(kFALSE);
  lego->SetPlaneColor(kBlue-5);
  lego->SetFontColor(kGray);
  lego->Set2DMode(TEveCaloLego::kValSize);
  lego->SetName("Calo Detail");
  gEve->AddElement(lego);

  // overlay lego

  TEveLegoOverlay* overlay = new TEveLegoOverlay();
  overlay->SetCaloLego(lego);
  v->AddOverlayElement(overlay);
  gEve->AddElement(overlay);

  // overlay legend

  ButtFaker* legend = new ButtFaker(v);
  v->AddOverlayElement(legend);

  gEve->Redraw3D(kTRUE);
}

#endif
