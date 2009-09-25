// @(#)root/eve:$Id: calorimeters.C 26568 2008-12-01 20:55:50Z matevz $
// Author: Alja Mrak-Tadel

// Demonstrates usage of EVE calorimetry classes.

#include "TEveProjections.h"

const char* histFile = "http://amraktad.web.cern.ch/amraktad/cms_calo_hist.root";

void calorimeters(Bool_t hdata = kTRUE)
{
   gSystem->IgnoreSignal(kSigSegmentationViolation, true);

   TFile::SetCacheFileDir(".");
   TEveManager::Create();
   gEve->GetSelection()->SetPickToSelect(1);

   // event data
   TFile* hf = TFile::Open(histFile, "CACHEREAD");
   TH2F* ecalHist = (TH2F*)hf->Get("ecalLego");
   TH2F* hcalHist = (TH2F*)hf->Get("hcalLego");
   TEveCaloData* data = 0;
   if (hdata)
   {
      TEveCaloDataHist* hd = new TEveCaloDataHist();
      Int_t slice;
      slice = hd->AddHistogram(ecalHist);
      hd->RefSliceInfo(slice).Setup("ECAL", 0.3, 41);
      slice = hd->AddHistogram(hcalHist);
      hd->RefSliceInfo(slice).Setup("HCAL", 0.1, 46);
      data = hd;
   }
   else
   {
      data = MakeVecData(ecalHist, hcalHist);
   }
   // axis attrib
   data->GetEtaBins()->SetTitleFont(120);
   data->GetEtaBins()->SetTitle("h");
   data->GetPhiBins()->SetTitleFont(120);
   data->GetPhiBins()->SetTitle("f");


   // window mng
   TEveWindowSlot  *slot  = 0;
   TEveWindowFrame *frame = 0;
   TEveViewer *v = 0;
   slot = TEveWindow::CreateWindowInTab(gEve->GetBrowser()->GetTabRight());
   TEveWindowPack* packH = slot->MakePack();
   packH->SetShowTitleBar(kFALSE);
   packH->SetHorizontal();

   {
      // 3D calorimeter
      TEveCalo3D* calo3d = MakeCalo3D(data, gEve->GetDefaultGLViewer(), gEve->GetEventScene());

      // projected
      slot = packH->NewSlot();
      TEveWindowPack* pack1 = slot->MakePack();
      // RPhi
      slot = pack1->NewSlot();
      v = new TEveViewer("RPhi-Viewer");
      v->SpawnGLEmbeddedViewer();
      slot->ReplaceWindow(v);
      gEve->GetViewers()->AddElement(v);
      TEveScene*  sP = gEve->SpawnNewScene("Projected");
      v->AddScene(sP);
      MakeCalo2D(calo3d, v->GetGLViewer(), sP, TEveProjection::kPT_RPhi);
      // RhoZ
      slot = pack1->NewSlot();
      v = new TEveViewer("RhoZ-Viewer");
      v->SpawnGLEmbeddedViewer();
      slot->ReplaceWindow(v);
      gEve->GetViewers()->AddElement(v);
      TEveScene*  s = gEve->SpawnNewScene("RhoZ");
      v->AddScene(s);
      MakeCalo2D(calo3d, v->GetGLViewer(), s, TEveProjection::kPT_RhoZ);
   }
   {
      // root canvas
      slot = packH->NewSlot();
      TEveWindowPack* pack2 = slot->MakePack();
      pack2->SetShowTitleBar(kFALSE);
      slot = pack2->NewSlot();
      slot->StartEmbedding();
      TCanvas* can = new TCanvas("Root Canvas");
      can->ToggleEditor();
      can->cd();
      THStack *hs = new THStack("hs","Stacked 1D histograms");
      ecalHist->SetFillColor(data->GetSliceColor(0));
      hcalHist->SetFillColor(data->GetSliceColor(1));
      hs->Add(ecalHist);;
      hs->Add(hcalHist);
      hs->Draw();
      slot->StopEmbedding();
   }
   {
      // calorimeter lego
      slot = pack2->NewSlot();
      v = new TEveViewer("LegoViewer");
      v->SpawnGLViewer(gEve->GetEditor());
      slot->ReplaceWindow(v);
      gEve->GetViewers()->AddElement(v);
      TEveScene*  sL = gEve->SpawnNewScene("Projected");
      v->AddScene(sL);
      MakeCaloLego(data, v->GetGLViewer(), sL);
   }

   gEve->GetBrowser()->GetTabRight()->SetTab(1);
   gEve->Redraw3D(1);
}

//______________________________________________________________________________
void MakeCaloLego(TEveCaloData* data, TGLViewer* v, TEveScene* s)
{
   // eta-phi histogram

   // histogram
   TEveCaloLego* lego = new TEveCaloLego(data);
   s->AddElement(lego);

   gStyle->SetPalette(1, 0);
   Bool_t usePalette = kFALSE;
   lego->Set2DMode(TEveCaloLego::kValSize);

   // lego->SetAutoRebin(kFALSE);
   lego->SetName("Calorimeter Lego");
   lego->SetPixelsPerBin(8);
   lego->InitMainTrans();
   Float_t sc = TMath::TwoPi();
   lego->RefMainTrans().SetScale(sc, sc, sc);

   // add overlay lego draws scales in 2D
   TEveCaloLegoOverlay* overlay = new TEveCaloLegoOverlay();
   v->AddOverlayElement(overlay);
   overlay->SetCaloLego(lego);

   v->SetCurrentCamera(TGLViewer::kCameraOrthoXOY);
   v->SetEventHandler(new TEveLegoEventHandler(v->GetGLWidget(), v, lego));
   gEve->AddToListTree(lego, kTRUE);
}

//______________________________________________________________________________
TEveCalo3D* MakeCalo3D(TEveCaloData* data, TGLViewer* v, TEveScene *s )
{
   // 3D towers

   TEveCalo3D* calo3d = new TEveCalo3D(data);
   calo3d->SetName("Calorimter 3D");
   calo3d->SetBarrelRadius(129);
   calo3d->SetEndCapPos(300);
   s->AddElement(calo3d);

   gEve->AddToListTree(calo3d, kTRUE);

   return calo3d;
}

//______________________________________________________________________________
void MakeCalo2D(TEveCalo3D* calo3d, TGLViewer *v, TEveScene *s, TEveProjection::EPType_e t)
{
   // projected calorimeter

   v->SetCurrentCamera(TGLViewer::kCameraOrthoXOY);
   //   v->SetGuideState(TGLUtil::kAxesOrigin, kTRUE, kFALSE, 0);

   TEveProjectionManager* mng = new TEveProjectionManager();
   mng->SetProjection(t);

   TEveProjectionAxes* axes = new TEveProjectionAxes(mng);
   s->AddElement(axes);
   TEveCalo2D* calo2d = (TEveCalo2D*) mng->ImportElements(calo3d);
   s->AddElement(calo2d);

   gEve->AddToListTree(mng, kTRUE);
   gEve->AddToListTree(calo2d, kTRUE);
}

//______________________________________________________________________________
TEveCaloDataVec* MakeVecData(TH2* h1, TH2* h2)
{
   // Example how to fill data when bins can be iregular.

  TEveCaloDataVec* data = new TEveCaloDataVec(2);

  data->RefSliceInfo(0).Setup("ECAL", 0.3, kRed);
  data->RefSliceInfo(1).Setup("HCAL", 0.1, kYellow);

  TAxis *ax =  h1->GetXaxis();
  TAxis *ay =  h1->GetYaxis();
  for(Int_t i=1; i<=ax->GetNbins(); i++)
  {
    for(Int_t j=1; j<=ay->GetNbins(); j++)
    {
      data->AddTower(ax->GetBinLowEdge(i), ax->GetBinUpEdge(i),
		     ay->GetBinLowEdge(j), ay->GetBinUpEdge(j));

      data->FillSlice(0, h1->GetBinContent(i, j));
      data->FillSlice(1, h2->GetBinContent(i, j));
    }
  }

  data->SetAxisFromBins();
  data->DataChanged();

  return data;
}
