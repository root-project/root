#include "TEveProjections.h"

const char* histFile = "http://amraktad.web.cern.ch/amraktad/cms_calo_hist.root";

void cms_calo()
{
  TFile::SetCacheFileDir(".");
  TEveManager::Create();
  gEve->GetSelection()->SetPickToSelect(1);
  
  // event data
  TFile* hf = TFile::Open(histFile, "CACHEREAD");
  TH2F* ecalHist = (TH2F*)hf->Get("ecalLego");
  TH2F* hcalHist = (TH2F*)hf->Get("hcalLego");
  TEveCaloDataHist* data = new TEveCaloDataHist();
  data->AddHistogram(ecalHist);
  data->AddHistogram(hcalHist);
  
  // different calorimeter presentations
  TEveCalo3D* calo3d = MakeCalo3D(data);
  MakeCalo2D(calo3d);
  MakeCaloLego(data);

  gEve->Redraw3D(1);
}

//______________________________________________________________________________
TEveCalo3D* MakeCalo3D(TEveCaloDataHist* data)
{
  // palette
  gStyle->SetPalette(1, 0);
  TEveRGBAPalette* pal = new TEveRGBAPalette(0, 100);
  pal->SetLimits(0, TMath::CeilNint(data->GetMaxVal()));
  pal->SetDefaultColor((Color_t)4);
  pal->SetShowDefValue(kFALSE);

  // 3D towers
  TEveCalo3D* calo3d = new TEveCalo3D(data);
  calo3d->SetBarrelRadius(129);
  calo3d->SetEndCapPos(300);
  calo3d->SetPalette(pal);
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
  mng->ImportElements(calo3d);
}

//______________________________________________________________________________
void MakeCaloLego(TEveCaloDataHist* data)
{
  TEveViewer* v2 = gEve->SpawnNewViewer("Lego Viewer");
  TGLViewer*  v  = v2->GetGLViewer();
  v->SetCurrentCamera(TGLViewer::kCameraPerspXOY);
  v->SetEventHandler(new TEveLegoEventHandler("Lego", v->GetGLWidget(), v));
  TEveRGBAPalette* pal = new TEveRGBAPalette(0, 100);
  pal->SetLimits(0, TMath::CeilNint(data->GetMaxVal()));
  pal->SetDefaultColor((Color_t)4);
  TEveScene*  s2 = gEve->SpawnNewScene("Lego");
  v2->AddScene(s2);
  
  // lego1
  TEveCaloLego* lego = new TEveCaloLego(data);
  lego->SetPalette(pal);
  lego->SetGridColor(kGray+2);
  lego->Set2DMode(TEveCaloLego::kValSize);
  lego->SetName("TwoHistLego");
  gEve->AddElement(lego, s2);
  gEve->AddToListTree(lego, kTRUE);

  // overlay lego1
  gEve->DisableRedraw();
  TEveLegoOverlay* overlay = new TEveLegoOverlay();
  overlay->SetCaloLego(lego);
  v->AddOverlayElement(overlay);
  gEve->AddElement(overlay, s2);
  gEve->EnableRedraw();

  // lego2
  TEveCaloDataHist* data2 = new TEveCaloDataHist();
  data2->AddHistogram(data->GetHistogram(0));
  TEveCaloLego* lego2 = new TEveCaloLego(data2);
  lego2->SetName("OneHistLego");
  lego2->SetPalette(pal);
  gEve->AddElement(lego2, s2);
  gEve->AddToListTree(lego2, kTRUE);  
  lego2->InitMainTrans();
  lego2->RefMainTrans().Move3PF(0, 0, 15);
  lego2->SetEta(-3, 3);
  lego2->SetPhi(TMath::PiOver4());
}
