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

  // palette
  gStyle->SetPalette(1, 0);

  // different calorimeter presentations
  TEveCaloData* data = 0;
  if (hdata)
  {
    TEveCaloDataHist* hd = new TEveCaloDataHist();
    Int_t s;
    s = hd->AddHistogram(ecalHist);
    hd->RefSliceInfo(s).Setup("ECAL", 0.3, kRed);
    s = hd->AddHistogram(hcalHist);
    hd->RefSliceInfo(s).Setup("HCAL", 0.1, kYellow);
    data = hd;
  }
  else
  {
    data = MakeVecData(ecalHist, hcalHist);
  }

  TEveCalo3D* calo3d = MakeCalo3D(data);
  MakeCalo2D(calo3d);
  MakeCaloLego(data);
  gEve->Redraw3D(1);
}

//______________________________________________________________________________
TEveCalo3D* MakeCalo3D(TEveCaloData* data)
{

  // 3D towers
  TEveCalo3D* calo3d = new TEveCalo3D(data);
  calo3d->SetBarrelRadius(129);
  calo3d->SetEndCapPos(300);
  gEve->AddElement(calo3d);

  return calo3d;
}

//______________________________________________________________________________
void MakeCalo2D(TEveCalo3D* calo3d)
{
  TEveViewer* v1 = gEve->SpawnNewViewer("2D Viewer");
  TEveScene*  s1 = gEve->SpawnNewScene("Projected Event");
  v1->AddScene(s1);
  TGLViewer* v = v1->GetGLViewer();
  v->SetCurrentCamera(TGLViewer::kCameraOrthoXOY);
  TGLCameraMarkupStyle* mup = v->GetCameraMarkup();
  if(mup) mup->SetShow(kFALSE);
  v->SetGuideState(TGLUtil::kAxesOrigin, kTRUE, kFALSE, 0);
  v->SetClearColor(kBlue + 4);

  // projected calorimeter
  TEveProjectionManager* mng = new TEveProjectionManager();
  mng->SetProjection(TEveProjection::kPT_RhoZ);
  gEve->AddElement(mng, s1);
  gEve->AddToListTree(mng, kTRUE);
  TEveCalo2D* c2d = (TEveCalo2D*) mng->ImportElements(calo3d);
  c2d->SetValueIsColor(kFALSE);
}

//______________________________________________________________________________
void MakeCaloLego(TEveCaloData* data)
{
   TEveViewer* v2 = gEve->SpawnNewViewer("Lego Viewer");
   TGLViewer*  v  = v2->GetGLViewer();
   v->SetCurrentCamera(TGLViewer::kCameraPerspXOY);
   v->SetEventHandler(new TEveLegoEventHandler("Lego", v->GetGLWidget(), v));
   TEveScene*  s2 = gEve->SpawnNewScene("Lego");
   v2->AddScene(s2);

   // lego
   TEveCaloLego* lego = new TEveCaloLego(data);
   lego->SetPlaneColor(kBlue-5);
   lego->Set2DMode(TEveCaloLego::kValSize);
   lego->SetName("TwoHistLego");
   gEve->AddElement(lego, s2);
   gEve->AddToListTree(lego, kTRUE);

   lego->InitMainTrans();
   Float_t sc = TMath::TwoPi();
   lego->RefMainTrans().SetScale(sc, sc, sc);
   // overlay lego1
   TEveLegoOverlay* overlay = new TEveLegoOverlay();
   v->AddOverlayElement(overlay);
   overlay->SetCaloLego(lego);

   TGLCameraOverlay* camInfo = new TGLCameraOverlay();
   camInfo->SetShowPerspective(kFALSE);
   v->AddOverlayElement(camInfo);
   gEve->AddElement(overlay, s2);
}

//______________________________________________________________________________
TEveCaloDataVec* MakeVecData(TH2* h1, TH2* h2)
{
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

  data->SetEtaBins(new TAxis(100, ax->GetBinLowEdge(1), ax->GetBinUpEdge(ax->GetLast())));
  data->SetPhiBins(new TAxis(100, ay->GetBinLowEdge(1), ay->GetBinUpEdge(ay->GetLast())));
  data->DataChanged();

  return data;
}
