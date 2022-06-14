#include "TCanvas.h"
#include "TFile.h"
#include "TH2F.h"
#include "TIterator.h"
#include "TKey.h"
#include "TLegend.h"
#include "TList.h"
#include "TMVA/probas.h"
#include "TMVA/tmvaglob.h"
#include "TString.h"

#include <iostream>

using std::cout;
using std::endl;

// this macro plots the MVA probability distributions (Signal and
// Background overlayed) of different MVA methods run in TMVA
// (e.g. running TMVAnalysis.C).

// input: - Input file (result from TMVA)
//        - use of TMVA plotting TStyle
void TMVA::probas(TString dataset, TString fin , Bool_t useTMVAStyle  )
{
   // set style and remove existing canvas'
   TMVAGlob::Initialize( useTMVAStyle );

   // switches
   const Bool_t Draw_CFANN_Logy = kFALSE;
   const Bool_t Save_Images     = kTRUE;

   // checks if file with name "fin" is already open, and if not opens one
   TFile* file = TMVAGlob::OpenFile( fin );  

   const Int_t width = 600;   // size of canvas

   // this defines how many canvases we need
   TCanvas *c = 0;

   // counter variables
   Int_t countCanvas = 0;

   // list of existing MVAs
   //const Int_t nveto = 1;
   TString suffixSig = "_tr_S";
   TString suffixBgd = "_tr_B";

   // search for the right histograms in full list of keys
   TList methods;
   UInt_t nmethods = TMVAGlob::GetListOfMethods( methods,file->GetDirectory(dataset.Data()) );
   if (nmethods==0) {
      cout << "--- Probas.C: no methods found!" << endl;
      return;
   }
   TIter next(&methods);
   TKey *key, *hkey;
   char fname[200];
   TH1* sig(0);
   TH1* bgd(0);
   

   while ( (key = (TKey*)next()) ) {
      TDirectory * mDir = (TDirectory*)key->ReadObj();
      TList titles;
      UInt_t ni = TMVAGlob::GetListOfTitles( mDir, titles );
      TString methodName;
      TMVAGlob::GetMethodName(methodName,key);
      if (ni==0) {
         cout << "+++ No titles found for classifier: " << methodName << endl;
         return;
      }
      TIter nextTitle(&titles);
      TKey *instkey;
      TDirectory *instDir;
      
      // iterate over all classifiers
      while ( (instkey = (TKey *)nextTitle()) ) {
         instDir = (TDirectory *)instkey->ReadObj();
         TString instName = instkey->GetName();
         TList h1hists;
         UInt_t nhists = TMVAGlob::GetListOfKeys( h1hists, "TH1", instDir );
         if (nhists==0) cout << "*** No histograms found!" << endl;
         TIter nextInDir(&h1hists);
         TString methodTitle;
         TMVAGlob::GetMethodTitle(methodTitle,instDir);
         Bool_t found = kFALSE;
         while ( (hkey = (TKey*)nextInDir()) ) {
            TH1 *th1 = (TH1*)hkey->ReadObj();
            TString hname= th1->GetName();
            if (hname.Contains( suffixSig ) && !hname.Contains( "Cut") && 
                !hname.Contains("original") && !hname.Contains("smoothed")) {
               // retrieve corresponding signal and background histograms   
               TString hnameS = hname;
               TString hnameB = hname; hnameB.ReplaceAll("_S","_B");

               sig = (TH1*)instDir->Get( hnameS );
               bgd = (TH1*)instDir->Get( hnameB );

               if (sig == 0 || bgd == 0) {
                  cout << "*** probas.C: big troubles in probas.... histogram: " << hname << " not found" << endl;
                  return;
               }

               TH1* sigF(0);
               TH1* bkgF(0);
               
               for (int i=0; i<= 5; i++) {
                  TString hspline = hnameS + Form("_smoothed_hist_from_spline%i",i);
                  sigF = (TH1*)instDir->Get( hspline );
            
                  if (sigF) {
                     bkgF = (TH1*)instDir->Get( hspline.ReplaceAll("_tr_S","_tr_B") );
                     break;
                  }
               }
               if (!sigF){
                  TString hspline = hnameS + TString("_smoothed_hist_from_KDE");
                  sigF = (TH1*)instDir->Get( hspline );
                  
                  if (sigF) {
                     bkgF = (TH1*)instDir->Get( hspline.ReplaceAll("_tr_S","_tr_B") );
                  }
               }
              
               if ((sigF == NULL || bkgF == NULL) &&!hname.Contains("hist") ) {
                  cout << "*** probas.C: big troubles - did not find probability histograms" << endl;
                  return;
               }
               else  {
                  // remove the signal suffix

                  // check that exist
                  if (NULL != sigF && NULL != bkgF && NULL!=sig && NULL!=bgd) {
          
                     found = kTRUE;
                     // chop off useless stuff
                     sig->SetTitle( TString("TMVA output for classifier: ") + methodTitle );
            
                     // create new canvas
                     cout << "--- Book canvas no: " << countCanvas << endl;
                     char cn[20];
                     sprintf( cn, "canvas%d", countCanvas+1 );
                     c = new TCanvas( cn, Form("TMVA Output Fit Variables %s",methodTitle.Data()), 
                                      countCanvas*50+200, countCanvas*20, width, width*0.78 ); 
            
                     // set the histogram style
                     TMVAGlob::SetSignalAndBackgroundStyle( sig, bgd );
                     TMVAGlob::SetSignalAndBackgroundStyle( sigF, bkgF );
            
                     // frame limits (choose judicuous x range)
                     Float_t nrms = 4;
                     Float_t xmin = TMath::Max( TMath::Min(sig->GetMean() - nrms*sig->GetRMS(), 
                                                           bgd->GetMean() - nrms*bgd->GetRMS() ),
                                                sig->GetXaxis()->GetXmin() );
                     Float_t xmax = TMath::Min( TMath::Max(sig->GetMean() + nrms*sig->GetRMS(), 
                                                           bgd->GetMean() + nrms*bgd->GetRMS() ),
                                                sig->GetXaxis()->GetXmax() );
                     Float_t ymin = 0;
                     Float_t ymax = TMath::Max( sig->GetMaximum(), bgd->GetMaximum() )*1.5;
            
                     if (Draw_CFANN_Logy && methodName == "CFANN") ymin = 0.01;
            
                     // build a frame
                     Int_t nb = 500;
                     TH2F* frame = new TH2F( TString("frame") + sig->GetName() + "_proba", sig->GetTitle(), 
                                             nb, xmin, xmax, nb, ymin, ymax );
                     frame->GetXaxis()->SetTitle(methodTitle);
                     frame->GetYaxis()->SetTitle("Normalized");
                     TMVAGlob::SetFrameStyle( frame );
            
                     // eventually: draw the frame
                     frame->Draw();  
            
                     if (Draw_CFANN_Logy && methodName == "CFANN") c->SetLogy();
            
                     // overlay signal and background histograms
                     sig->SetMarkerColor( TMVAGlob::getSignalLine() );
                     sig->SetMarkerSize( 0.7 );
                     sig->SetMarkerStyle( 20 );
                     sig->SetLineWidth(1);

                     bgd->SetMarkerColor( TMVAGlob::getBackgroundLine() );
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
                     sprintf( fname, "%s/plots/mva_pdf_%s_c%i",dataset.Data(), methodTitle.Data(), countCanvas+1 );
                     if (Save_Images) TMVAGlob::imgconv( c, fname );
                     countCanvas++;
                  }
               }
            }
            
         }
         if(!found){
            cout << "--- No PDFs found for method " << methodTitle << ". Did you request \"CreateMVAPdfs\" in the option string?" << endl;
         }
      }    
   }
}
