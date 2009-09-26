// @(#)root/eve:$Id: calorimeters.C 26568 2008-12-01 20:55:50Z matevz $
// Author: Alja Mrak-Tadel

// Demonstrates usage of EVE calorimetry classes.

#include "TEveProjections.h"

const char* histFile = "http://amraktad.web.cern.ch/amraktad/cms_calo_hist.root";

void calorimeters()
{
   gSystem->IgnoreSignal(kSigSegmentationViolation, true);

   TFile::SetCacheFileDir(".");
   TEveManager::Create();
   gEve->GetSelection()->SetPickToSelect(1);

   // event data
   TFile* hf = TFile::Open(histFile, "CACHEREAD");
   TH2F* ecalHist = (TH2F*)hf->Get("ecalLego");
   TH2F* hcalHist = (TH2F*)hf->Get("hcalLego");

   TEveCaloDataHist* data = new TEveCaloDataHist();
   data->AddHistogram(ecalHist);
   data->RefSliceInfo(0).Setup("ECAL", 0.3, 41);
   data->AddHistogram(hcalHist);
   data->RefSliceInfo(1).Setup("HCAL", 0.1, 46);

   // axis attrib
   data->GetEtaBins()->SetTitleFont(120);
   data->GetEtaBins()->SetTitle("h");
   data->GetPhiBins()->SetTitleFont(120);
   data->GetPhiBins()->SetTitle("f");

   MakeCaloLego(data);

   BuildProjectedView(data);

   gEve->Redraw3D(1);
}

//______________________________________________________________________________
void BuildProjectedView(TEveCaloData* data)
{
   // Create a new tab with 3D calorimiter
   //  and projected (RPhi/RhoZ) calorimiter.

   TEveWindowSlot* slot;

   slot = TEveWindow::CreateWindowInTab(gEve->GetBrowser()->GetTabRight());
   TEveWindowPack* packH = slot->MakePack();
   packH->SetElementName("Projections");
   packH->SetHorizontal();
   packH->SetShowTitleBar(kFALSE);

   TEveWindowSlot* slotLeft = packH->NewSlot();
   slot = packH->NewSlot();

   TEveWindowPack* pack1 = slot->MakePack();
   TEveWindowSlot* slotRightTop = pack1->NewSlot();
   slotRightBottom = pack1->NewSlot();

   // build scenes
   TEveCalo3D* calo3d = MakeCalo3D(data, slotRightTop);
   MakeCalo2D(calo3d, slotLeft, TEveProjection::kPT_RPhi);
   MakeCalo2D(calo3d, slotRightBottom, TEveProjection::kPT_RhoZ);
}

//______________________________________________________________________________
void MakeCaloLego(TEveCaloData* data)
{
   // Eta-phi lego view.

   TGLViewer* v = gEve->GetDefaultGLViewer();
   TEveScene* s = gEve->GetEventScene();
   gStyle->SetPalette(1, 0);

   TEveCaloLego* lego = new TEveCaloLego(data);
   s->AddElement(lego);
   lego->Set2DMode(TEveCaloLego::kValSize);

   // lego->SetAutoRebin(kFALSE);
   lego->SetName("Calorimeter Lego");
   lego->SetPixelsPerBin(8);
   lego->InitMainTrans();
   Float_t sc = TMath::TwoPi();
   lego->RefMainTrans().SetScale(sc, sc, sc);

   // draws scales
   TEveCaloLegoOverlay* overlay = new TEveCaloLegoOverlay();
   v->AddOverlayElement(overlay);
   overlay->SetCaloLego(lego);

   v->SetCurrentCamera(TGLViewer::kCameraOrthoXOY);
   v->SetEventHandler(new TEveLegoEventHandler(v->GetGLWidget(), v, lego));
   gEve->AddToListTree(lego, kTRUE);
}

//______________________________________________________________________________
TEveCalo3D* MakeCalo3D(TEveCaloData* data, TEveWindowSlot* slot)
{
   // 3D catersian view. 

   TEveViewer* v = new TEveViewer("3D Viewer");
   v->SpawnGLEmbeddedViewer();
   slot->ReplaceWindow(v);
   gEve->GetViewers()->AddElement(v);
   TEveScene*  s = gEve->SpawnNewScene("3D Scene");
   v->AddScene(s);

   TEveCalo3D* calo3d = new TEveCalo3D(data);
   calo3d->SetName("Calorimter 3D");
   calo3d->SetBarrelRadius(129);
   calo3d->SetEndCapPos(300);
   s->AddElement(calo3d);

   gEve->AddToListTree(calo3d, kTRUE);
   return calo3d;
}

//______________________________________________________________________________
void MakeCalo2D(TEveCalo3D* calo3d, TEveWindowSlot* slot, TEveProjection::EPType_e t)
{
   // Projected calorimeter.

   TEveViewer* v = new TEveViewer("2D Viewer");
   v->SpawnGLEmbeddedViewer();
   slot->ReplaceWindow(v);
   gEve->GetViewers()->AddElement(v);
   TEveScene*  s = gEve->SpawnNewScene("Scene");
   v->AddScene(s);

   TEveProjectionManager* mng = new TEveProjectionManager();
   mng->SetProjection(t);

   TEveProjectionAxes* axes = new TEveProjectionAxes(mng);
   s->AddElement(axes);
   TEveCalo2D* calo2d = (TEveCalo2D*) mng->ImportElements(calo3d);
   s->AddElement(calo2d);

   v->GetGLViewer()->SetCurrentCamera(TGLViewer::kCameraOrthoXOY);

   gEve->AddToListTree(mng, kTRUE);
   gEve->AddToListTree(calo2d, kTRUE);
}
