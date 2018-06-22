/// \file
/// \ingroup tutorial_eve
/// Demonstrates usage of EVE calorimetry classes.
///
/// \image html eve_calorimeters.png
/// \macro_code
///
/// \author Alja Mrak-Tadel

#include "TEveProjections.h"
#include "TSystem.h"
#include "TEveCalo.h"
#include "TEveWindow.h"
#include "TEveManager.h"
#include "TEveBrowser.h"
#include "TEveProjectionAxes.h"
#include "TEveScene.h"
#include "TEveViewer.h"
#include "TEveTrans.h"
#include "TEveCaloLegoOverlay.h"
#include "TEveLegoEventHandler.h"
#include "TEveGedEditor.h"
#include "TEveJetCone.h"

#include "TGLWidget.h"
#include "TGLViewer.h"
#include "TGTab.h"

#include "TFile.h"
#include "TAxis.h"


const char* histFile =
   "http://amraktad.web.cern.ch/amraktad/cms_calo_hist.root";

TEveCaloLego* MakeCaloLego(TEveCaloData* data, TEveWindowSlot* slot);
TEveCalo2D* MakeCalo2D(TEveCalo3D* calo3d, TEveWindowSlot* slot, TEveProjection::EPType_e t);
TEveCalo3D* MakeCalo3D(TEveCaloData* data, TEveWindowSlot* slot);
TEveCalo2D* MakeCalo2D(TEveCalo3D* calo3d, TEveWindowSlot* slot, TEveProjection::EPType_e t);
void MakeViewerScene(TEveWindowSlot* slot, TEveViewer*& v, TEveScene*& s);
void add_jet(TEveElement*, const char*, Float_t, Float_t, Float_t, Float_t);

void calorimeters()
{
   // gSystem->IgnoreSignal(kSigSegmentationViolation, true);
   TEveManager::Create();

   // event data
   TFile::SetCacheFileDir(".");
   auto hf = TFile::Open(histFile, "CACHEREAD");
   auto ecalHist = (TH2F*)hf->Get("ecalLego");
   auto hcalHist = (TH2F*)hf->Get("hcalLego");
   auto data = new TEveCaloDataHist();
   data->AddHistogram(ecalHist);
   data->RefSliceInfo(0).Setup("ECAL", 0.3, kBlue);
   data->AddHistogram(hcalHist);
   data->RefSliceInfo(1).Setup("HCAL", 0.1, kRed);
   data->GetEtaBins()->SetTitleFont(120);
   data->GetEtaBins()->SetTitle("h");
   data->GetPhiBins()->SetTitleFont(120);
   data->GetPhiBins()->SetTitle("f");
   data->IncDenyDestroy();
   gEve->AddToListTree(data, kFALSE);


   // first tab
   auto lego = MakeCaloLego(data, 0);

   //
   // second tab
   //

   // frames
   auto slot = TEveWindow::CreateWindowInTab(gEve->GetBrowser()->GetTabRight());
   auto packH = slot->MakePack();
   packH->SetElementName("Projections");
   packH->SetHorizontal();
   packH->SetShowTitleBar(kFALSE);

   slot = packH->NewSlot();
   auto pack0 = slot->MakePack();
   pack0->SetShowTitleBar(kFALSE);
   auto  slotLeftTop   = pack0->NewSlot();
   auto slotLeftBottom = pack0->NewSlot();

   slot = packH->NewSlot();
   auto pack1 = slot->MakePack();
   pack1->SetShowTitleBar(kFALSE);
   auto slotRightTop    = pack1->NewSlot();
   auto slotRightBottom = pack1->NewSlot();

   // viewers ans scenes in second tab
   auto calo3d = MakeCalo3D(data, slotRightTop);
   MakeCalo2D(calo3d, slotLeftTop, TEveProjection::kPT_RPhi);
   MakeCalo2D(calo3d, slotLeftBottom, TEveProjection::kPT_RhoZ);
   lego = MakeCaloLego(data, slotRightBottom);


   gEve->GetBrowser()->GetTabRight()->SetTab(1);
   gEve->Redraw3D(kTRUE);
}

//______________________________________________________________________________
TEveCaloLego* MakeCaloLego(TEveCaloData* data, TEveWindowSlot* slot)
{
   // Eta-phi lego view.

   TEveViewer* v;
   TEveScene* s;
   if (slot) {
      MakeViewerScene(slot, v, s);
   } else {
      v = gEve->GetDefaultViewer();
      s = gEve->GetEventScene();
   }
   v->SetElementName("Viewer - Lego");
   s->SetElementName("Scene - Lego");

   auto lego = new TEveCaloLego(data);
   s->AddElement(lego);

   // By the default lego extends is (1x1x1). Resize it to put in 'natural'
   // coordinates, so that y extend in 2*Pi and set height of lego two times
   //  smaller than y extend to have better view in 3D perspective.
   lego->InitMainTrans();
   lego->RefMainTrans().SetScale(TMath::TwoPi(), TMath::TwoPi(), TMath::Pi());

   // draws scales and axis on borders of window
   auto glv = v->GetGLViewer();
   TEveCaloLegoOverlay* overlay = new TEveCaloLegoOverlay();
   glv->AddOverlayElement(overlay);
   overlay->SetCaloLego(lego);

   // set event handler to move from perspective to orthographic view.
   glv->SetCurrentCamera(TGLViewer::kCameraOrthoXOY);
   glv->SetEventHandler
      (new TEveLegoEventHandler(glv->GetGLWidget(), glv, lego));
   gEve->AddToListTree(lego, kTRUE);

   return lego;
}

//______________________________________________________________________________
TEveCalo3D* MakeCalo3D(TEveCaloData* data, TEveWindowSlot* slot)
{
   // 3D cartesian view.

   TEveViewer* v; TEveScene* s;
   MakeViewerScene(slot, v, s);
   v->SetElementName("Viewer - 3D");
   s->SetElementName("Scene - 3D");

   auto calo3d = new TEveCalo3D(data);
   calo3d->SetBarrelRadius(129.00);
   calo3d->SetEndCapPos(268.36);
   s->AddElement(calo3d);

   add_jet(calo3d, "JetCone Lojz",  1.4,  1.0, 0.4, 0.2);
   add_jet(calo3d, "JetCone Mici", -2.0, -2.1, 0.2, 0.4);

   return calo3d;
}

//______________________________________________________________________________
TEveCalo2D* MakeCalo2D(TEveCalo3D* calo3d, TEveWindowSlot* slot,
                       TEveProjection::EPType_e t)
{
   // Projected calorimeter.

   TEveViewer* v; TEveScene* s;
   MakeViewerScene(slot, v, s);
   v->SetElementName("Viewer - 2D");
   s->SetElementName("Scene - 2D");

   auto mng = new TEveProjectionManager();
   mng->SetProjection(t);

   auto axes = new TEveProjectionAxes(mng);
   s->AddElement(axes);
   TEveCalo2D* calo2d = (TEveCalo2D*) mng->ImportElements(calo3d);
   s->AddElement(calo2d);

   v->GetGLViewer()->SetCurrentCamera(TGLViewer::kCameraOrthoXOY);

   gEve->AddToListTree(mng, kTRUE);
   gEve->AddToListTree(calo2d, kTRUE);

   return calo2d;
}

//______________________________________________________________________________
void MakeViewerScene(TEveWindowSlot* slot, TEveViewer*& v, TEveScene*& s)
{
   // Create a scene and a viewer in the given slot.

   v = new TEveViewer("Viewer");
   v->SpawnGLViewer(gEve->GetEditor());
   slot->ReplaceWindow(v);
   gEve->GetViewers()->AddElement(v);
   s = gEve->SpawnNewScene("Scene");
   v->AddScene(s);
}

//______________________________________________________________________________
void add_jet(TEveElement* parent, const char* name,
             Float_t eta, Float_t phi,
             Float_t deta, Float_t dphi)
{
   auto jet = new TEveJetCone(name, name);
   jet->SetMainTransparency(60);
   jet->SetLineColor(kRed);
   jet->SetCylinder(129 - 10, 268.36 - 10);
   jet->AddEllipticCone(eta, phi, deta, dphi);
   jet->SetPickable(kTRUE);
   jet->SetHighlightFrame(kFALSE);
   parent->AddElement(jet);
}
