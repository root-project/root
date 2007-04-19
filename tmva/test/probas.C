#include "tmvaglob.C"

// this macro plots the resulting MVA distribution (Signal and
// Background overlayed) of different MVA methods run in TMVA
// (e.g. running TMVAnalysis.C).

// input: - Input file (result from TMVA)
//        - use of TMVA plotting TStyle
void probas( TString fin = "TMVA.root", Bool_t useTMVAStyle = kTRUE )
{
   cout << "---PROBAS.C" << endl;
   // set style and remove existing canvas'
   TMVAGlob::Initialize( useTMVAStyle );

   // switches
   const Bool_t Draw_CFANN_Logy = kFALSE;
   const Bool_t Save_Images     = kTRUE;

   // checks if file with name "fin" is already open, and if not opens one
   TFile* file = TMVAGlob::OpenFile( fin );  

   // define Canvas layout here!
   Int_t xPad = 1; // no of plots in x
   Int_t yPad = 1; // no of plots in y
   Int_t noPad = xPad * yPad ; 
   const Int_t width = 600;   // size of canvas

   // this defines how many canvases we need
   TCanvas *c = 0;

   // counter variables
   Int_t countCanvas = 0;

   // list of existing MVAs
   const Int_t nveto = 1;
   TString suffixSig = "_tr_S";
   TString suffixBgd = "_tr_B";

   // search for the right histograms in full list of keys
   TIter next(file->GetListOfKeys());
   TKey *key, *hkey;
   char fname[200];
   TH1* sig(0);
   TH1* bgd(0);
   while ((key = (TKey*)next())) {

      if (TString(key->GetClassName()) != "TDirectory" && TString(key->GetClassName()) != "TDirectoryFile") continue;
      if (!TString(key->GetName()).BeginsWith("Method_")) continue;
     
      TDirectory * mDir = (TDirectory*)key->ReadObj();
      TIter nextInMDir(mDir->GetListOfKeys());
      while (hkey = (TKey*)nextInMDir()) {

         // make sure, that we only look at histograms
         TClass *cl = gROOT->GetClass(hkey->GetClassName());
         if (!cl->InheritsFrom("TH1")) continue;
         TH1 *th1 = (TH1*)hkey->ReadObj();
         TString hname= th1->GetName();

         if (hname.Contains( suffixSig ) && !hname.Contains( "Cut")) {
            // retrieve corresponding signal and background histograms   
            TString remove="tr";
            if(hname.Contains( "tr" )&&!hname.Contains("hist"))
               { 
                  sig = (TH1*)mDir->Get( hname  );
                  bgd = (TH1*)mDir->Get( hname.ReplaceAll("_S","_B"));
               }
            hname.ReplaceAll("_B","_S");
            TH1* sigF(0);
            TH1* bkgF(0);
            for (int i=0; i<= 5; i++) {
               TString hspline = hname +Form("_hist_from_spline%i",i);
               sigF = (TH1*)mDir->Get( hspline );
  	    
               if (sigF) {
                  bkgF = (TH1*)mDir->Get( hspline.ReplaceAll("_tr_S","_tr_B") );
              
                  break;
               }
            }
            cout << sig->GetName() << endl;
            if ((sigF == NULL || bkgF == NULL) &&!hname.Contains("hist") ) {
               cout << "--- probas.C: did not find spline for histogram: " << hname.Data() << sigF << " " << bkgF << endl;
            }
            else  {

               // remove the signal suffix

               // check that exist
               if (NULL != sigF && NULL != bkgF && NULL!=sig && NULL!=bgd) {
          

                  TString hname = sig->GetName();
            
                  // chop off useless stuff
                  TString title(sig->GetTitle());
                  title.ReplaceAll("_tr_S","");
                  sig->SetTitle( TString("MVA output for method: ") + title );
            
                  // create new canvas
                  cout << "--- Book canvas no: " << countCanvas << endl;
                  char cn[20];
                  sprintf( cn, "canvas%d", countCanvas+1 );
                  c = new TCanvas( cn, Form("MVA Output Fit Variables %s",title.Data()), 
                                   countCanvas*50+200, countCanvas*20, width, width*0.78 ); 
            
                  // set the histogram style
                  TMVAGlob::SetSignalAndBackgroundStyle( sig, bgd );
                  TMVAGlob::SetSignalAndBackgroundStyle( sigF, bkgF );
            
                  // frame limits (choose judicuous x range)
                  Float_t nrms = 4;
                  cout << "--- mean and RMS (S): " << sig->GetMean() << ", " << sig->GetRMS() << endl;
                  cout << "--- mean and RMS (B): " << bgd->GetMean() << ", " << bgd->GetRMS() << endl;
                  Float_t xmin = TMath::Max( TMath::Min(sig->GetMean() - nrms*sig->GetRMS(), 
                                                        bgd->GetMean() - nrms*bgd->GetRMS() ),
                                             sig->GetXaxis()->GetXmin() );
                  Float_t xmax = TMath::Min( TMath::Max(sig->GetMean() + nrms*sig->GetRMS(), 
                                                        bgd->GetMean() + nrms*bgd->GetRMS() ),
                                             sig->GetXaxis()->GetXmax() );
                  Float_t ymin = 0;
                  Float_t ymax = TMath::Max( sig->GetMaximum(), bgd->GetMaximum() )*1.5;
            
                  if (Draw_CFANN_Logy && mvaName[imva] == "CFANN") ymin = 0.01;
            
                  // build a frame
                  Int_t nb = 500;
                  TH2F* frame = new TH2F( TString("frame") + sig->GetName(), sig->GetTitle(), 
                                          nb, xmin, xmax, nb, ymin, ymax );
                  frame->GetXaxis()->SetTitle(title);
                  frame->GetYaxis()->SetTitle("Normalized");
                  TMVAGlob::SetFrameStyle( frame );
            
                  // eventually: draw the frame
                  frame->Draw();  
            
                  if (Draw_CFANN_Logy && mvaName[imva] == "CFANN") c->SetLogy();
            
                  // overlay signal and background histograms
                  sig->SetMarkerColor(4);
                  sig->SetMarkerSize( 0.7 );
                  sig->SetMarkerStyle( 20 );
                  sig->SetLineWidth(1);

                  bgd->SetMarkerColor(2);
                  bgd->SetMarkerSize( 0.7 );
                  bgd->SetMarkerStyle( 24 );
                  bgd->SetLineWidth(1);

                  sig->Draw("samee");
                  bgd->Draw("samee");
            
                  sigF->SetFillStyle( 0 );
                  bkgF->SetFillStyle( 0 );
                  sigF->Draw("samehist");
                  bkgF->Draw("samehist");
            
                  // redraw axes
                  frame->Draw("sameaxis");
            
                  // Draw legend               
                  TLegend *legend= new TLegend( c->GetLeftMargin(), 1 - c->GetTopMargin() - 0.2, 
                                                c->GetLeftMargin() + 0.4, 1 - c->GetTopMargin() );
                  legend->AddEntry(sig,"Signal data","P");
                  legend->AddEntry(sigF,"Signal PDF","L");
                  legend->AddEntry(bgd,"Background data","P");
                  legend->AddEntry(bkgF,"Background PDF","L");
                  legend->Draw("same");
                  legend->SetBorderSize(1);
                  legend->SetMargin( 0.3 );
            
                  // save canvas to file
                  c->Update();
                  TMVAGlob::plot_logo();
                  sprintf( fname, "plots/mva_c%i", countCanvas+1 );
                  if (Save_Images) TMVAGlob::imgconv( c, fname );
                  countCanvas++;
               }
            }
         }
      }
   }
}
