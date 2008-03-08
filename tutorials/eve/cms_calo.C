#include "TEveProjections.h"

const char* histFile = "http://amraktad.web.cern.ch/amraktad/cms_calo_hist.root";

void cms_calo()
{
   TFile::SetCacheFileDir(".");
   TEveManager::Create();
   gEve->GetSelection()->SetPickToSelect(1);
   gEve->GetHighlight()->SetPickToSelect(0);
   TGLViewer* v1 = gEve->GetGLViewer();
   v1->SetGuideState(TGLUtil::kAxesOrigin, kTRUE, kFALSE, 0);
   // palette
   gStyle->SetPalette(1, 0);
   TEveRGBAPalette* pal = new TEveRGBAPalette(0, 100);
   pal->SetLimits(0, 50);
   pal->SetDefaultColor((Color_t)4);
   pal->SetShowDefValue(kFALSE);

   // calorimeter
   TFile* hf = TFile::Open(histFile, "CACHEREAD");
   TH2F* hcalHist = (TH2F*)hf->Get("hcalLego");
   TH2F* ecalHist = (TH2F*)hf->Get("ecalLego");
   TEveCaloDataHist* data = new TEveCaloDataHist();
   data->AddHistogram(ecalHist);
   data->AddHistogram(hcalHist);
   TEveCalo3D* calo = new TEveCalo3D(data);
   calo->SetBarrelRadius(129);
   calo->SetEndCapPos(300);
   calo->SetPalette(pal);
   gEve->AddElement(calo);

   // projections
   TEveViewer* nv = gEve->SpawnNewViewer("Projected");
   TEveScene*  ns = gEve->SpawnNewScene("Projected Event");
   nv->AddScene(ns);
   TGLViewer* v = nv->GetGLViewer();
   v->SetCurrentCamera(TGLViewer::kCameraOrthoXOY);
   TGLCameraMarkupStyle* mup = v->GetCameraMarkup();
   if(mup) mup->SetShow(kFALSE);
   v->SetGuideState(TGLUtil::kAxesOrigin, kTRUE, kFALSE, 0);
   v->SetClearColor(kBlue + 4);

   TEveProjectionManager* mng = new TEveProjectionManager();
   mng->SetProjection(TEveProjection::kPT_RhoZ);
   gEve->AddElement(mng, ns);
   gEve->AddToListTree(mng, kTRUE);

   mng->ImportElements(calo);

   gEve->Redraw3D(1);
}
