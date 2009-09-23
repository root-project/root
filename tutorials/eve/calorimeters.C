// @(#)root/eve:$Id: triangleset.C 26568 2008-12-01 20:55:50Z matevz $
// Author: Alja Mrak-Tadel

// Demonstrates usage of EVE calorimetry classes.

#include "TEveProjections.h"

const char* histFile = "http://amraktad.web.cern.ch/amraktad/cms_calo_hist.root";

void cms_calo(Bool_t hdata = kTRUE)
{
   gSystem->IgnoreSignal(kSigSegmentationViolation, true);

   TFile::SetCacheFileDir(".");
   TEveManager::Create();
   gEve->GetSelection()->SetPickToSelect(1);

   // event data
   TFile* hf = TFile::Open(histFile, "CACHEREAD");
   TH2F* ecalHist = (TH2F*)hf->Get("ecalLego");
   TH2F* hcalHist = (TH2F*)hf->Get("hcalLego");

   // fill data
   TEveCaloData* data = 0;
   if (hdata)
   {
      TEveCaloDataHist* hd = new TEveCaloDataHist();
      Int_t slice;
      slice = hd->AddHistogram(ecalHist);
      hd->RefSliceInfo(slice).Setup("ECAL", 0.3, kRed);
      slice = hd->AddHistogram(hcalHist);
      hd->RefSliceInfo(slice).Setup("HCAL", 0.1, kBlue);
      data = hd;
   }
   else
   {
      data = MakeVecData(ecalHist, hcalHist);
   }
   // set eta, phi axis title with symbol.ttf font
   data->GetEtaBins()->SetTitleFont(120);
   data->GetEtaBins()->SetTitle("h");
   data->GetPhiBins()->SetTitleFont(120);
   data->GetPhiBins()->SetTitle("f");

   // different calorimeter presentations
   TEveViewer* v = (TEveViewer*)gEve->GetViewers()->FirstChild();
   TEveScene* s = (TEveScene*)gEve->GetScenes()->FirstChild();
   MakeCaloLego(data, v, s);

   TEveViewer* v3D = gEve->SpawnNewViewer("3D Viewer");
   TEveScene*  s3D = gEve->SpawnNewScene("3D scene");
   v3D->AddScene(s3D);
   TEveCalo3D* calo3d = MakeCalo3D(data, v3D, s3D);

   TEveViewer* vP = gEve->SpawnNewViewer("2D Viewer");
   TEveScene*  sP = gEve->SpawnNewScene("Projected Event");
   vP->AddScene(sP);
   MakeCalo2D(calo3d, vP, sP);

   gEve->Redraw3D(1);
}

//______________________________________________________________________________
void MakeCaloLego(TEveCaloData* data, TEveViewer* ev, TEveScene* s)
{
   // eta-phi histogram

   TGLViewer*  v  = ev->GetGLViewer();

   // lego
   TEveCaloLego* lego = new TEveCaloLego(data);
   s->AddElement(lego);
   lego->Set2DMode(TEveCaloLego::kValSize);
   lego->SetName("TwoHistLego");
   lego->SetPixelsPerBin(8);
   lego->InitMainTrans();
   Float_t sc = TMath::TwoPi();
   lego->RefMainTrans().SetScale(sc, sc, sc);
   lego->SetAutoRebin(kFALSE);

   // add overlay lego draws scales in 2D
   TEveCaloLegoOverlay* overlay = new TEveCaloLegoOverlay();
   overlay->SetShowPlane(kTRUE);
   v->AddOverlayElement(overlay);
   overlay->SetCaloLego(lego);

   v->SetCurrentCamera(TGLViewer::kCameraOrthoXOY);
   v->SetEventHandler(new TEveLegoEventHandler(lego, v->GetGLWidget(), v));
   gEve->AddToListTree(lego, kTRUE);
}

//______________________________________________________________________________
TEveCalo3D* MakeCalo3D(TEveCaloData* data, TEveViewer *v, TEveScene *s )
{
   // 3D towers

   TEveCalo3D* calo3d = new TEveCalo3D(data);
   calo3d->SetBarrelRadius(129);
   calo3d->SetEndCapPos(300);
   s->AddElement(calo3d);

   gEve->AddToListTree(calo3d, kTRUE);

   return calo3d;
}

//______________________________________________________________________________
void MakeCalo2D(TEveCalo3D* calo3d, TEveViewer *ev, TEveScene *s)
{
   // projected calorimeter

   TGLViewer* v = ev->GetGLViewer();
   v->SetCurrentCamera(TGLViewer::kCameraOrthoXOY);
   v->SetGuideState(TGLUtil::kAxesOrigin, kTRUE, kFALSE, 0);
   v->ColorSet().Background().SetColor(kBlue + 4);

   TEveProjectionManager* mng = new TEveProjectionManager();
   mng->SetProjection(TEveProjection::kPT_RhoZ);

   TEveProjectionAxes* axes = new TEveProjectionAxes(mng);
   axes->SetTitle("TEveProjections demo");
   s->AddElement(axes);
   TEveCalo2D* calo2d = (TEveCalo2D*) mng->ImportElements(calo3d);
   s->AddElement(calo2d);

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
