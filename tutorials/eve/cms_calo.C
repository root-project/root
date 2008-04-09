#include "TEveProjections.h"

const char* histFile = "http://amraktad.web.cern.ch/amraktad/cms_calo_hist.root";

void cms_calo()
{
  TFile::SetCacheFileDir(".");
  TEveManager::Create();
  gEve->GetSelection()->SetPickToSelect(1);
  //  gEve->GetHighlight()->SetPickToSelect(0);
  
  // event data
  TFile* hf = TFile::Open(histFile, "CACHEREAD");
  TH2F* hcalHist = (TH2F*)hf->Get("hcalLego");
  TH2F* ecalHist = (TH2F*)hf->Get("ecalLego");
  TEveCaloDataHist* data = new TEveCaloDataHist();
  data->AddHistogram(ecalHist);
  data->AddHistogram(hcalHist);

  Double_t etaLimLow  = hcalHist->GetXaxis()->GetXmin();
  Double_t etaLimHigh = hcalHist->GetXaxis()->GetXmax();

  // palette
  gStyle->SetPalette(1, 0);
  TEveRGBAPalette* pal = new TEveRGBAPalette(0, 100);
  pal->SetLimits(0, data->GetMaxVal());
  pal->SetDefaultColor((Color_t)4);
  pal->SetShowDefValue(kFALSE);

  // calo 3D
  TEveCalo3D* calo3d = new TEveCalo3D(data);
  calo3d->SetBarrelRadius(129);
  calo3d->SetEndCapPos(300);
  calo3d->SetEtaLimits(etaLimLow, etaLimHigh);
  calo3d->SetPalette(pal);
  gEve->AddElement(calo3d);

  // calo 2D
  TEveViewer* v1 = gEve->SpawnNewViewer("Projected 2D");
  TEveScene*  s1 = gEve->SpawnNewScene("Projected Event");
  v1->AddScene(s1);
  TGLViewer* v = v1->GetGLViewer();
  v->SetCurrentCamera(TGLViewer::kCameraOrthoXOY);
  TGLCameraMarkupStyle* mup = v->GetCameraMarkup();
  if(mup) mup->SetShow(kFALSE);
  v->SetGuideState(TGLUtil::kAxesOrigin, kTRUE, kFALSE, 0);
  v->SetClearColor(kBlue + 4);

  TEveProjectionManager* mng = new TEveProjectionManager();
  mng->SetProjection(TEveProjection::kPT_RhoZ);
  gEve->AddElement(mng, s1);
  gEve->AddToListTree(mng, kTRUE);
  mng->ImportElements(calo3d);


  // lego
  TEveRGBAPalette* pal = new TEveRGBAPalette(0, 100);
  pal->SetLimits(0, data->GetMaxVal());
  pal->SetDefaultColor((Color_t)4);
  TEveViewer* v2 = gEve->SpawnNewViewer("Lego ");
  v2->GetGLViewer()->SetCurrentCamera(TGLViewer::kCameraPerspXOY);
  TEveScene*  s2 = gEve->SpawnNewScene("Lego");
  v2->AddScene(s2);
  TEveCaloLego* lego = new TEveCaloLego(data);
  lego->SetPalette(pal);
  lego->SetEtaLimits(etaLimLow, etaLimHigh);
  lego->SetTitle("caloTower Et distribution");
  lego->SetGridColor(kOrange - 8);
  lego->InitMainTrans();
  lego->RefMainTrans().RotateLF(1, 2, -TMath::PiOver2());
  gEve->AddElement(lego, s2);
  gEve->AddToListTree(lego, kTRUE);

  gEve->Redraw3D(1);
}
