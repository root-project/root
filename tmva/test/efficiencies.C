#include "tmvaglob.C"



void plot_efficiencies( TFile* file, Int_t type = 2 , TDirectory* BinDir)
{
   // input:   - Input file (result from TMVA),
   //          - type = 1 --> plot efficiency(B) versus eff(S)
   //                 = 2 --> plot rejection (B) versus efficiency (S)

   Bool_t __PRINT_LOGO__ = kTRUE;
   Bool_t __SAVE_IMAGE__ = kTRUE;

   //   cout << "Bindir=" << BinDir->GetName() << endl;

   // the coordinates
   Float_t x1 = 0;
   Float_t x2 = 1;
   Float_t y1 = 0;
   Float_t y2 = 0.5;

   // reverse order if "rejection"
   if (type == 2) {
      Float_t z = y1;
      y1 = 1 - y2;
      y2 = 1 - z;    
      //      cout << "--- type==2: plot background rejection versus signal efficiency" << endl;
   }
   else {
      //  cout << "--- type==1: plot background efficiency versus signal efficiency" << endl;
   }
   // create canvas
   TCanvas* c = new TCanvas( "c", "the canvas", 200, 0, 650, 500 );
   c->SetBorderMode(0);
   c->SetFillColor(10);

   // global style settings
   gPad->SetGrid();
   gPad->SetTicks();

   // legend
   Float_t x0L = 0.107,     y0H = 0.899;
   Float_t dxL = 0.457-x0L, dyH = 0.22;
   if (type == 2) {
      x0L = 0.15;
      y0H = 1 - y0H + dyH + 0.07;
   }
   TLegend *legend = new TLegend( x0L, y0H-dyH, x0L+dxL, y0H );
   legend->SetBorderSize(1);
   legend->SetTextSize( 0.05 );
   legend->SetHeader( "MVA Method:" );
   legend->SetMargin( 0.4 );
   legend->SetFillColor(0);

   TString xtit = "Signal efficiency";
   TString ytit = "Background efficiency";  
   if (type == 2) ytit = "Background rejection";
   TString ftit = ytit + " versus " + xtit;

   if (TString(BinDir->GetName()).Contains("multicut")){
      ftit += "  Bin: ";
      ftit += (BinDir->GetTitle());
   }
   // draw empty frame
   if(gROOT->FindObject("frame")!=0) gROOT->FindObject("frame")->Delete();
   TH2F* frame = new TH2F( "frame", ftit, 500, x1, x2, 500, y1, y2 );
   frame->GetXaxis()->SetTitle( xtit );
   frame->GetYaxis()->SetTitle( ytit );
   TMVAGlob::SetFrameStyle( frame, 1.0 );

   frame->Draw();  

   Int_t color = 1;
   Int_t nmva  = 0;
   TIter next(file->GetListOfKeys());
   TKey *key;

   TString hNameRef = "effBvsS";
   if (type == 2) hNameRef = "rejBvsS";

   TList hists;

   // loop over all histograms with that name
   while (key = (TKey*)next()) {
      TClass *cl = gROOT->GetClass(key->GetClassName());
      if (!cl->InheritsFrom("TH1")) continue;    
      TH1 *h = (TH1*)key->ReadObj();    
      TString hname = h->GetName();
      
      if (hname.Contains( hNameRef ) && hname.BeginsWith( "MVA_" )) {
         h->SetLineWidth(3);
         h->SetLineColor(color);
         color++; if (color == 5 || color == 10 || color == 11) color++; 
         h->Draw("csame");
         hists.Add(h);
         nmva++;
      }
   }

   while (hists.GetSize()) {
      TListIter hIt(&hists);
      TH1* hist(0);
      Double_t largestInt=0;
      TH1* histWithLargestInt(0);
      while ((hist = (TH1*)hIt())!=0) {
         Double_t integral = hist->Integral(1,hist->FindBin(0.9999));
         if (integral>largestInt) {
            largestInt = integral;
            histWithLargestInt = hist;
         }
      }
      legend->AddEntry(histWithLargestInt,TString(histWithLargestInt->GetTitle()).ReplaceAll("MVA_",""),"l");
      hists.Remove(histWithLargestInt);
   }   
   
   // rescale legend box size
   // current box size has been tuned for 3 MVAs + 1 title
   if (type == 1) {
      dyH *= (1.0 + Float_t(nmva - 3.0)/4.0);
      legend->SetY1( y0H - dyH );
   }
   else {
      dyH *= (Float_t(nmva - 3.0)/4.0);
      legend->SetY2( y0H + dyH);
   }

   // redraw axes
   frame->Draw("sameaxis");  
   legend->Draw("same");

   // ============================================================

   if (__PRINT_LOGO__) TMVAGlob::plot_logo();

   // ============================================================

   c->Update();

   TString fname = "plots/" + hNameRef;
   if (TString(BinDir->GetName()).Contains("multicut")){
      TString fprepend(BinDir->GetName());
      fprepend.ReplaceAll("multicutMVA_","");
      fname = "plots/" + fprepend + "_" + hNameRef;
   }
   if (__SAVE_IMAGE__) TMVAGlob::imgconv( c, fname );

   return;
}

void efficiencies( TString fin = "TMVA.root", Int_t type = 2, Bool_t useTMVAStyle = kTRUE )
{
   // argument: type = 1 --> plot efficiency(B) versus eff(S)
   //           type = 2 --> plot rejection (B) versus efficiency (S)
  
   // set style and remove existing canvas'
   TMVAGlob::Initialize( useTMVAStyle );

   // checks if file with name "fin" is already open, and if not opens one
   TFile* file = TMVAGlob::OpenFile( fin );  

   // check if multi-cut MVA or only one set of MVAs
   Bool_t multiMVA=kFALSE;
   TIter nextDir(file->GetListOfKeys());
   TKey *key;
   // loop over all directories and check if
   // one contains the key word 'multicutMVA'
   while ((key = (TKey*)nextDir())) {
      TClass *cl = gROOT->GetClass(key->GetClassName());
      if (!cl->InheritsFrom("TDirectory")) continue;    
      TDirectory *d = (TDirectory*)key->ReadObj();    
      TString path(d->GetPath());
      if ((TString(d->GetPath())).Contains("multicutMVA")){         
         multiMVA=kTRUE;
         plot_efficiencies( file, type, d );
      }
   }
   plot_efficiencies( file, type, gDirectory );

   return;
}

