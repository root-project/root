// @(#)root/eve:$Id: calorimeters.C 26568 2008-12-01 20:55:50Z matevz $
// Author: Alja Mrak-Tadel

// Demonstrates usage of EVE calorimetry classes.

#include "TEveProjections.h"

const char* histFile = "http://amraktad.web.cern.ch/amraktad/cms_calo_hist.root";

void calorimeters()
{
   gSystem->IgnoreSignal(kSigSegmentationViolation, true);
   TEveManager::Create();

   // event data
   TFile::SetCacheFileDir(".");
   TFile* hf = TFile::Open(histFile, "CACHEREAD");
   TH2F* ecalHist = (TH2F*)hf->Get("ecalLego");
   TH2F* hcalHist = (TH2F*)hf->Get("hcalLego");
   TEveCaloDataHist* data = new TEveCaloDataHist();
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
   TEveCaloLego* lego = MakeCaloLego(data, 0);

   //
   // second tab
   //

   // frames
   TEveWindowSlot* slot = TEveWindow::CreateWindowInTab(gEve->GetBrowser()->GetTabRight());
   TEveWindowPack* packH = slot->MakePack();
   packH->SetElementName("Projections");
   packH->SetHorizontal();
   packH->SetShowTitleBar(kFALSE);

   slot = packH->NewSlot();
   TEveWindowPack* pack0 = slot->MakePack();
   pack0->SetShowTitleBar(kFALSE);
   TEveWindowSlot*  slotLeftTop   = pack0->NewSlot();
   TEveWindowSlot* slotLeftBottom = pack0->NewSlot();

   slot = packH->NewSlot();
   TEveWindowPack* pack1 = slot->MakePack();
   pack1->SetShowTitleBar(kFALSE);
   TEveWindowSlot* slotRightTop    = pack1->NewSlot();
   TEveWindowSlot* slotRightBottom = pack1->NewSlot();

   // viewers ans scenes in second tab
   TEveCalo3D* calo3d = MakeCalo3D(data, slotRightTop);
   MakeCalo2D(calo3d, slotLeftTop, TEveProjection::kPT_RPhi);
   MakeCalo2D(calo3d, slotLeftBottom, TEveProjection::kPT_RhoZ);
   TEveCaloLego* lego = MakeCaloLego(data, slotRightBottom);


   gEve->GetBrowser()->GetTabRight()->SetTab(1);
   gEve->Redraw3D(kTRUE);
}

//______________________________________________________________________________
TEveCaloLego* MakeCaloLego(TEveCaloData* data, TEveWindowSlot* slot)
{
   // Eta-phi lego view.

   TEveViewer* v;
   TEveScene* s;
   if (slot)
   {
      TEveViewer* v; TEveScene* s;
      MakeViewerScene(slot, v, s);
   } else {
      v = gEve->GetDefaultViewer();
      s = gEve->GetEventScene();
   }
   v->SetElementName("Viewer - Lego");
   s->SetElementName("Scene - Lego");

   gStyle->SetPalette(1, 0);
   TEveCaloLego* lego = new TEveCaloLego(data);
   s->AddElement(lego);

   // move to real world coordinates
   lego->InitMainTrans();
   Float_t sc = TMath::Min(lego->GetEtaRng(), lego->GetPhiRng());
   lego->RefMainTrans().SetScale(sc, sc, sc);

   // draws scales and axis on borders of window
   TGLViewer* glv = v->GetGLViewer();
   TEveCaloLegoOverlay* overlay = new TEveCaloLegoOverlay();
   glv->AddOverlayElement(overlay);
   overlay->SetCaloLego(lego);

   // set event handler to move from perspective to orthographic view.
   glv->SetCurrentCamera(TGLViewer::kCameraOrthoXOY);
   glv->SetEventHandler(new TEveLegoEventHandler(glv->GetGLWidget(), glv, lego));
   gEve->AddToListTree(lego, kTRUE);

   return lego;
}

//______________________________________________________________________________
TEveCalo3D* MakeCalo3D(TEveCaloData* data, TEveWindowSlot* slot)
{
   // 3D catersian view.

   TEveViewer* v; TEveScene* s;
   MakeViewerScene(slot, v, s);
   v->SetElementName("Viewer - 3D");
   s->SetElementName("Scene - 3D");

   TEveCalo3D* calo3d = new TEveCalo3D(data);
   calo3d->SetBarrelRadius(129);
   calo3d->SetEndCapPos(300);
   s->AddElement(calo3d);

   return calo3d;
}

//______________________________________________________________________________
TEveCalo2D* MakeCalo2D(TEveCalo3D* calo3d, TEveWindowSlot* slot, TEveProjection::EPType_e t)
{
   // Projected calorimeter.

   TEveViewer* v; TEveScene* s;
   MakeViewerScene(slot, v, s);
   v->SetElementName("Viewer - 2D");
   s->SetElementName("Scene - 2D");

   TEveProjectionManager* mng = new TEveProjectionManager();
   mng->SetProjection(t);

   TEveProjectionAxes* axes = new TEveProjectionAxes(mng);
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
