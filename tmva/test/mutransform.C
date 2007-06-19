#include "tmvaglob.C"

// this macro plots the mu-transformation results of the MVA output
// variables from the various MVA methods run in TMVA (e.g. running
// TMVAnalysis.C). Since the mu-transform is flat for background, only
// signal is shown. The more signal ispeaked towards one, the better is 
// the discrimination of the MVA method. See this reference for more 
// information (in French): 
//  http://tel.archives-ouvertes.fr/documents/archives0/00/00/29/91/index_fr.html
//
// input: - Input file (result from TMVA),
//        - use log/lin scale
//        - use of TMVA plotting TStyle
void mutransform( TString fin = "TMVA.root", Bool_t logy = kFALSE, Bool_t useTMVAStyle = kTRUE  )
{
   // set style and remove existing canvas'
   TMVAGlob::Initialize( useTMVAStyle );

   // checks if file with name "fin" is already open, and if not opens one
   TFile* file = TMVAGlob::OpenFile( fin );  

   // the coordinates
   Float_t x1 = 0.0;
   Float_t x2 = 1;
   Float_t y1 = 0;
   Float_t y2 = -1;
   Float_t ys = 1.08;

   enum { nskip = 1 };
   TString hskip[nskip] = { "Variable" };

   // loop over all histograms with that name
   // search for maximum ordinate
   TIter   next(file->GetListOfKeys());
   TKey   *key, *hkey;
   Int_t   nmva  = 0;
   TString hName = "muTransform_S"; // ignore background
   while (key = (TKey*)next()) {

      if (TString(key->GetClassName()) != "TDirectory" && TString(key->GetClassName()) != "TDirectoryFile") continue;
      if(! TString(key->GetName()).BeginsWith("Method_") ) continue;
      
      TDirectory * mDir = (TDirectory*)key->ReadObj();
      TIter nextInMDir(mDir->GetListOfKeys());
      while (hkey = (TKey*)nextInMDir()) {
         TClass *cl = gROOT->GetClass(hkey->GetClassName());
         if (!cl->InheritsFrom("TH1")) continue;
         TH1 *h = (TH1*)hkey->ReadObj();
         Bool_t skip = !TString(h->GetName()).Contains( hName );
         for (Int_t iskip=0; iskip<nskip; iskip++) 
            if (TString(h->GetName()).Contains( hskip[iskip] )) skip = kTRUE;
         if (!skip) {
            if (h->GetMaximum() > y2) y2 = h->GetMaximum()*ys;
            nmva++;
         }
      }
   }
   if (y2 == -1) {
      cout << "No mu-transforms found" << endl;
      return;
   }

   // create canvas
   TCanvas* c = new TCanvas( "c", "the canvas", 150, 0, 650, 500 );

   // global style settings
   c->SetTicks();
   if (logy) {
      y1 = 0.1;
      ys = 2.0;
      c->SetLogy();    
   }

   // legend
   Float_t x0L = 0.140, y0H = 0.86;
   Float_t dxL = 0.48,  dyH = 0.22;
   TLegend *legend = new TLegend( x0L, y0H-dyH, x0L+dxL, y0H );
   legend->SetTextSize( 0.05 );
   legend->SetHeader( "TMVA Classifier:" );
   legend->SetMargin( 0.4 );

   TString xtit = "mu-transform";
   TString ytit = "";  
   TString ftit = "Signal " + xtit;
  
   cout << "--- set frame maximum to: " << y2 << endl;
   next.Reset();

   // rescale legend box size
   // current box size has been tuned for 3 MVAs + 1 title
   dyH *= (1.0 + Float_t(nmva - 3.0)/4.0);
   legend->SetY1( y0H - dyH );

   // draw empty frame
   TH2F* frame = new TH2F( "frame", ftit, 500, x1, x2, 500, y1, y2 );
   frame->GetXaxis()->SetTitle( xtit );
   frame->GetYaxis()->SetTitle( ytit );
   TMVAGlob::SetFrameStyle( frame, 1.0 );

   frame->Draw();  

   // loop over all histograms with that name
   // plot
   Int_t color = 1;
   while (key = (TKey*)next()) {
      if( TString(key->GetClassName()) != "TDirectory" ) continue;
      if(! TString(key->GetName()).BeginsWith("Method_") ) continue;
      
      TDirectory * mDir = (TDirectory*)key->ReadObj();
      TIter nextInMDir(mDir->GetListOfKeys());
      while (hkey = (TKey*)nextInMDir()) {
         TClass *cl = gROOT->GetClass(hkey->GetClassName());
         if (!cl->InheritsFrom("TH1")) continue;
         TH1 *h = (TH1*)hkey->ReadObj();
         Bool_t skip = !TString(h->GetName()).Contains( hName );
         for (Int_t iskip=0; iskip<nskip; iskip++) 
            if (TString(h->GetName()).Contains( hskip[iskip] )) skip = kTRUE;
         if (!skip) {
            // signal or background ?
            if (TString(h->GetName()).Contains( "_S" )) {
               h->SetLineStyle( 1 );
               h->SetLineWidth( 3 );
            }
            else {
               h->SetLineStyle( 2 );
               h->SetLineWidth( 3 );
            }
            h->SetLineColor(color);
            color++;
            TString tit = h->GetTitle();
            tit.ReplaceAll( "mu-Transform", "" );
            tit.ReplaceAll( "(S)", "" );
            tit.ReplaceAll( ":", "" );
            legend->AddEntry( h, tit, "l" );
            h->Draw("same");
         }
      }
   }
  
   // redraw axes
   frame->Draw("sameaxis");

   legend->Draw("same");
   c->Update();

   TString fname = "plots/mutransform";
   TMVAGlob::imgconv( c, fname );
} 

