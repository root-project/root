#include "TMVA/compareanapp.h"
#include "TMVA/Types.h"
#include "TH2.h"


#define CheckDerivedPlots 0
//TString DerivedPlotName = "Proba";
TString DerivedPlotName = "Rarity";

void TMVA::compareanapp( TString finAn , TString finApp , 
                   HistType htype , bool useTMVAStyle )
{
   cout << "=== Compare histograms of two files ===" << endl;
   cout << "    File-1: " << finAn << endl;
   cout << "    File-2: " << finApp << endl;

   // set style and remove existing canvas'
   TMVAGlob::Initialize( useTMVAStyle );

   // switches
   const Bool_t Draw_CFANN_Logy = kFALSE;
   const Bool_t Save_Images     = kTRUE;

   TFile* file    = TMVAGlob::OpenFile( finAn );  
   TFile* fileApp = new TFile( finApp );
   file->cd();

   // define Canvas layout here!
   const Int_t width = 600;   // size of canvas

   // counter variables
   Int_t countCanvas = 0;

   TList methods;
   TIter next(&methods);
   TKey *key;   
   while ( (key = (TKey*)next()) ) {

      TString dirname = ((TDirectory*)key->ReadObj())->GetName();
      if (dirname.Contains( "Cuts" )) {
         cout << "--- Found directory: " << dirname << " --> ignoring" << endl;
         continue;
      }
      cout << "--- Found directory: " << dirname 
           << " --> going in" << endl;

      TString methodName;
      TMVAGlob::GetMethodName(methodName,key);

      cout << "--- Method: " << methodName << endl;

      //TDirectory* mDir = (TDirectory*)
      key->ReadObj();
      TList titles;
      //      UInt_t ninst = TMVAGlob::GetListOfTitles(mDir,titles);
      TIter nextTitle(&titles);
      TKey *titkey;
      TDirectory *titDir;
      while ( (titkey = TMVAGlob::NextKey(nextTitle,"TDirectory")) ) {

         titDir = (TDirectory *)titkey->ReadObj();
         TString methodTitle;
         TMVAGlob::GetMethodTitle(methodTitle,titDir);
         TString hname = "MVA_" + methodTitle;
         if (CheckDerivedPlots) hname += TString("_") + DerivedPlotName;

         TH1* sig = dynamic_cast<TH1*>(titDir->Get( hname + "_S" ));
         TH1* bgd = dynamic_cast<TH1*>(titDir->Get( hname + "_B" ));

         if (sig==0 || bgd==0) continue;

         // chop off useless stuff
         sig->SetTitle( Form("TMVA output for classifier: %s", methodTitle.Data()) );
         if      (htype == kProbaType) 
            sig->SetTitle( Form("TMVA probability for classifier: %s", methodTitle.Data()) );
         else if (htype == kRarityType) 
            sig->SetTitle( Form("TMVA Rarity for classifier: %s", methodTitle.Data()) );
         
         // create new canvas
         TString ctitle = ((htype == TMVA::kMVAType) ? 
                           Form("TMVA output %s",methodTitle.Data()) : 
                           (htype == kProbaType) ? 
                           Form("TMVA probability %s",methodTitle.Data()) :
                           Form("TMVA rarity %s",methodTitle.Data()));

         TString cname = ((htype == TMVA::kMVAType) ? 
                          Form("output_%s",methodTitle.Data()) : 
                          (htype == kProbaType) ? 
                          Form("probability_%s",methodTitle.Data()) :
                          Form("rarity_%s",methodTitle.Data()));

         TCanvas* c = new TCanvas( Form("canvas%d", countCanvas+1), ctitle, 
                          countCanvas*50+200, countCanvas*20, width, width*0.78 ); 
          
         // set the histogram style
         TMVAGlob::SetSignalAndBackgroundStyle( sig, bgd );
         
         // normalise both signal and background
         TMVAGlob::NormalizeHists( sig, bgd );
         
         // frame limits (choose judicuous x range)
         Float_t nrms = 4;
         Float_t xmin = TMath::Max( TMath::Min(sig->GetMean() - nrms*sig->GetRMS(), 
                                               bgd->GetMean() - nrms*bgd->GetRMS() ),
                                    sig->GetXaxis()->GetXmin() );
         Float_t xmax = TMath::Min( TMath::Max(sig->GetMean() + nrms*sig->GetRMS(), 
                                               bgd->GetMean() + nrms*bgd->GetRMS() ),
                                    sig->GetXaxis()->GetXmax() );
         Float_t ymin = 0;
         Float_t ymax = TMath::Max( sig->GetMaximum(), bgd->GetMaximum() )*1.2 ;
         
         if (Draw_CFANN_Logy && methodName == "CFANN") ymin = 0.01;
         
         // build a frame
         Int_t nb = 500;
         TH2F* frame = new TH2F( TString("frame") + methodTitle, sig->GetTitle(), 
                                 nb, xmin, xmax, nb, ymin, ymax );
         frame->GetXaxis()->SetTitle(methodTitle);
         if      (htype == kProbaType ) frame->GetXaxis()->SetTitle( "Signal probability" );
         else if (htype == kRarityType) frame->GetXaxis()->SetTitle( "Signal rarity" );
         frame->GetYaxis()->SetTitle("Normalized");
         TMVAGlob::SetFrameStyle( frame );
         
         // eventually: draw the frame
         frame->Draw();  
         
         c->GetPad(0)->SetLeftMargin( 0.105 );
         frame->GetYaxis()->SetTitleOffset( 1.2 );
         
         if (Draw_CFANN_Logy && methodName == "CFANN") c->SetLogy();
         
         // Draw legend               
         TLegend *legend= new TLegend( c->GetLeftMargin(), 1 - c->GetTopMargin() - 0.12, 
                                       c->GetLeftMargin() + 0.3, 1 - c->GetTopMargin() );
         legend->SetFillStyle( 1 );
         legend->AddEntry(sig,"Signal","F");
         legend->AddEntry(bgd,"Background","F");
         legend->SetBorderSize(1);
         legend->SetMargin( 0.3 );
         legend->Draw("same");

         // overlay signal and background histograms
         sig->Draw("samehist");
         bgd->Draw("samehist");

         // retrieve corresponding histogram from TMVApp.root   
         TString hStem(hname);
         cout << "--- Searching for histogram: " << hStem.Data() << " in application file" << endl;
         
         TH1* testHist = (TH1*)fileApp->Get( hStem );
         if (testHist != 0) {
            cout << "--> Found application histogram: " << testHist->GetName() << " --> superimpose it" << endl;
            // compute normalisation factor
            TMVAGlob::NormalizeHists( testHist );
            testHist->SetLineWidth( 3 );
            testHist->SetLineColor( 1 );
            testHist->Draw("samehist");
         }
         
         // redraw axes
         frame->Draw("sameaxis");
         
         // text for overflows
         Int_t    nbin = sig->GetNbinsX();
         Double_t dxu  = sig->GetBinWidth(0);
         Double_t dxo  = sig->GetBinWidth(nbin+1);
         TString uoflow = Form( "U/O-flow (S,B): (%.1f, %.1f)%% / (%.1f, %.1f)%%", 
                                sig->GetBinContent(0)*dxu*100, bgd->GetBinContent(0)*dxu*100,
                                sig->GetBinContent(nbin+1)*dxo*100, bgd->GetBinContent(nbin+1)*dxo*100 );
         TText* t = new TText( 0.975, 0.115, uoflow );
         t->SetNDC();
         t->SetTextSize( 0.030 );
         t->SetTextAngle( 90 );
         t->AppendPad();    
         
         // save canvas to file
         c->Update();
         TMVAGlob::plot_logo();
         if (Save_Images) {
            if      (htype == TMVA::kMVAType)   TMVAGlob::imgconv( c, Form("plots/mva_%s",    methodTitle.Data()) );
            else if (htype == TMVA::kProbaType) TMVAGlob::imgconv( c, Form("plots/proba_%s",  methodTitle.Data()) ); 
            else                         TMVAGlob::imgconv( c, Form("plots/rarity_%s", methodTitle.Data()) ); 
         }
         countCanvas++;
      }
   }
}
