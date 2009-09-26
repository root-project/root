// @(#)root/eve:$Id:  26568 2008-12-01 20:55:50Z matevz $
// Author: Alja Mrak-Tadel

// Calorimeter detailed view by using TEveCaloDataVec as data-source.
// Demonstrantes how to plot calorimiter data with irregular bins.

const char* histFile = "http://amraktad.web.cern.ch/amraktad/cms_calo_hist.root";

void cms_calo_detail()
{
   TEveManager::Create();

   // data
   TEveCaloDataVec* data = MakeVecData(10);

   // lego
   TEveCaloLego* lego = new TEveCaloLego(data);
   lego->SetElementName("Irregualar Lego");
   lego->SetAutoRebin(kFALSE);
   gEve->AddElement(lego);

   TGLViewer* v = gEve->GetDefaultGLViewer(); // Default
   v->SetCurrentCamera(TGLViewer::kCameraOrthoXOY);
   TEveLegoEventHandler* eh = new TEveLegoEventHandler((TGWindow*)v->GetGLWidget(), (TObject*)v, lego);
   v->SetEventHandler(eh);  

   lego->Set2DMode(TEveCaloLego::kValSize);
   gEve->Redraw3D(kTRUE);
}

//______________________________________________________________________________
TEveCaloDataVec* MakeVecData(Int_t ncells=0)
{
   // Example how to fill data when bins can be iregular.
   // If ncells = 0 (default) whole histogram is taken,
   // otherwise just ncells cells around the maximum.

   TFile::SetCacheFileDir(".");
   TFile* hf = TFile::Open(histFile, "CACHEREAD");
   TH2F* h1 = (TH2F*)hf->Get("ecalLego");
   TH2F* h2 = (TH2F*)hf->Get("hcalLego");

   TEveCaloDataVec* data = new TEveCaloDataVec(3);

   data->RefSliceInfo(0).Setup("ECAL", 0.3, kRed);
   data->RefSliceInfo(1).Setup("HCAL", 0.1, kYellow);
   data->RefSliceInfo(2).Setup("OTHER", 0, kCyan);

   TAxis *ax =  h1->GetXaxis();
   TAxis *ay =  h1->GetYaxis();

   Int_t xm = 1, xM = ax->GetNbins();
   Int_t ym = 1, yM = ay->GetNbins();
   if (ncells != 0)
   {
      Int_t cx, cy, cz;
      h1->GetMaximumBin(cx, cy, cz);
      xm = TMath::Max(xm, cx-ncells);
      xM = TMath::Min(xM, cx+ncells);
      ym = TMath::Max(ym, cy-ncells);
      yM = TMath::Min(yM, cy+ncells);
   }

   for(Int_t i=xm; i<=xM; i++)
   {
      for(Int_t j=ym; j<=yM; j++)
      {
         data->AddTower(ax->GetBinLowEdge(i), ax->GetBinUpEdge(i),
                        ay->GetBinLowEdge(j), ay->GetBinUpEdge(j));

         data->FillSlice(0, h1->GetBinContent(i, j));
         data->FillSlice(1, h2->GetBinContent(i, j));
      }
   }      

   // Add irregularities
   //
   Float_t off = 0.02;
   i = cx + 5; j = cy+2;
   data->AddTower(ax->GetBinLowEdge(i) -off, ax->GetBinUpEdge(i)+off,
                  ay->GetBinLowEdge(j) -off, ay->GetBinUpEdge(j)+off);
   data->FillSlice(2, 2.);

   i = cx-3; j = cy+3;
   data->AddTower(ax->GetBinLowEdge(i) -off, ax->GetBinUpEdge(i)+off,
                  ay->GetBinLowEdge(j) -off, ay->GetBinUpEdge(j)+off);
   data->FillSlice(2, 12.);
   
   data->DataChanged();
   data->SetAxisFromBins(0.001, 0.001);
   return data;
}

