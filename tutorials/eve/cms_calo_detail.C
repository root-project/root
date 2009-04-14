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
#include <TEveTrans.h>
#include <TEveCalo.h>
#include <TEveCaloData.h>
#include <TEveCaloLegoOverlay.h>
#include <TEveLegoEventHandler.h>

#include <TEveStraightLineSet.h>

#include <TGLViewer.h>
#include <TGLOverlayButton.h>

#include <TAxis.h>

#ifdef WIN32
#include <Windows4root.h>
#pragma comment(lib, "OpenGL32.lib")
#endif


class ButtFaker : public TGLOverlayButton
{
   ButtFaker(const ButtFaker&);            // Not implemented
   ButtFaker& operator=(const ButtFaker&); // Not implemented

public:
   TEveCaloLego* fLego;

   ButtFaker(TGLViewerBase *parent) :
      TGLOverlayButton(parent, "FlipColors", 10, 200, 80, 16),
      fLego(0)
   {
   }

   virtual ~ButtFaker() {}

   virtual void Clicked(TGLViewerBase*)
   {
      TEveCaloData* data = fLego->GetData();
      if (data->GetSliceColor(0) == kRed)
      {
         fLego->SetDataSliceColor(1, kRed);
         fLego->SetDataSliceColor(0, kYellow);
      }
      else
      {
         fLego->SetDataSliceColor(0, kRed);
         fLego->SetDataSliceColor(1, kYellow);
      }
      gEve->Redraw3D();
   } 

   ClassDef(ButtFaker,0);
};


void cms_calo_detail()
{
   TEveManager::Create();

   TGLViewer* v = gEve->GetDefaultGLViewer(); // Default
   v->SetCurrentCamera(TGLViewer::kCameraOrthoXOY);
   TEveLegoEventHandler* eh = new TEveLegoEventHandler("Lego", (TGWindow*)v->GetGLWidget(), (TObject*)v);
   eh->fMode = TEveLegoEventHandler::kLocked;
   v->SetEventHandler(eh);

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

   // lego
   TEveCaloLego* lego = new TEveCaloLego(data);
   // move to real-world coordinates
   Double_t em, eM, pm, pM;
   data->GetEtaLimits(em, eM);
   data->GetPhiLimits(pm, pM);
   lego->SetEta(em, eM);
   lego->SetPhiWithRng((pm+pM)*0.5, pM-pm);
   Double_t sc = ((eM - em) < (pM - pm)) ? (eM - em) : (pM - pm);
   lego->InitMainTrans();
   lego->RefMainTrans().SetScale(sc, sc, sc);
   lego->RefMainTrans().SetPos((eM+em)*0.5, (pM+pm)*0.5, 0);

   lego->SetAutoRebin(kFALSE);
   lego->SetPlaneColor(kBlue-5);
   lego->SetFontColor(kGray);
   lego->Set2DMode(TEveCaloLego::kValSize);
   lego->SetName("Calo Detail");
   gEve->AddElement(lego);

   // add line to test real world coordinates
   TEveStraightLineSet* ls = new TEveStraightLineSet();
   ls->AddLine( em, pm, 0,  eM, pM, 0);
   ls->SetLineColor(kBlue);
   gEve->AddElement(ls);

   // overlay lego
   TEveCaloLegoOverlay* overlay = new TEveCaloLegoOverlay();
   overlay->SetCaloLego(lego);
   v->AddOverlayElement(overlay);
   gEve->AddElement(overlay);

   // button overlay
   ButtFaker* legend = new ButtFaker(v);
   legend->fLego = lego;
   v->AddOverlayElement(legend);

   gEve->Redraw3D(kTRUE);
}

#endif
