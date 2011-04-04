#include "TLegend.h"
#include "TText.h"
#include "TH2.h"

#include "tmvaglob.C"

// this macro plots the resulting MVA distributions (Signal and
// Background overlayed) of different MVA methods run in TMVA
// (e.g. running TMVAnalysis.C).

enum HistType { MVAType = 0, ProbaType = 1, RarityType = 2, CompareType = 3 };

// input: - Input file (result from TMVA)
//        - use of TMVA plotting TStyle
void mvas( TString fin = "TMVA.root", HistType htype = MVAType, Bool_t useTMVAStyle = kTRUE )
{
   // set style and remove existing canvas'
   TMVAGlob::Initialize( useTMVAStyle );

   // switches
   const Bool_t Save_Images = kTRUE;

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

   // search for the right histograms in full list of keys
   TIter next(file->GetListOfKeys());
   TKey *key(0);   
   while ((key = (TKey*)next())) {

      if (!TString(key->GetName()).BeginsWith("Method_")) continue;
      if (!gROOT->GetClass(key->GetClassName())->InheritsFrom("TDirectory")) continue;

      TString methodName;
      TMVAGlob::GetMethodName(methodName,key);

      TDirectory* mDir = (TDirectory*)key->ReadObj();

      TIter keyIt(mDir->GetListOfKeys());
      TKey *titkey;
      while ((titkey = (TKey*)keyIt())) {

         if (!gROOT->GetClass(titkey->GetClassName())->InheritsFrom("TDirectory")) continue;

         TDirectory *titDir = (TDirectory *)titkey->ReadObj();
         TString methodTitle;
         TMVAGlob::GetMethodTitle(methodTitle,titDir);

         cout << "--- Found directory for method: " << methodName << "::" << methodTitle << flush;
         TString hname = "MVA_" + methodTitle;
         if      (htype == ProbaType  ) hname += "_Proba";
         else if (htype == RarityType ) hname += "_Rarity";
         TH1* sig = dynamic_cast<TH1*>(titDir->Get( hname + "_S" ));
         TH1* bgd = dynamic_cast<TH1*>(titDir->Get( hname + "_B" ));

         if (sig==0 || bgd==0) {
            if     (htype == MVAType)     
               cout << ":\t mva distribution not available (this is normal for Cut classifier)" << endl;
            else if(htype == ProbaType)   
               cout << ":\t probability distribution not available" << endl;
            else if(htype == RarityType)  
               cout << ":\t rarity distribution not available" << endl;
            else if(htype == CompareType) 
               cout << ":\t overtraining check not available" << endl;
            else cout << endl;
            continue;
         }

         cout << " containing " << hname << "_S/_B" << endl;
         // chop off useless stuff
         sig->SetTitle( Form("TMVA response for classifier: %s", methodTitle.Data()) );
         if      (htype == ProbaType) 
            sig->SetTitle( Form("TMVA probability for classifier: %s", methodTitle.Data()) );
         else if (htype == RarityType) 
            sig->SetTitle( Form("TMVA Rarity for classifier: %s", methodTitle.Data()) );
         else if (htype == CompareType) 
            sig->SetTitle( Form("TMVA overtraining check for classifier: %s", methodTitle.Data()) );
         
         // create new canvas
         TString ctitle = ((htype == MVAType) ? 
                           Form("TMVA response %s",methodTitle.Data()) : 
                           (htype == ProbaType) ? 
                           Form("TMVA probability %s",methodTitle.Data()) :
                           (htype == CompareType) ? 
                           Form("TMVA comparison %s",methodTitle.Data()) :
                           Form("TMVA Rarity %s",methodTitle.Data()));
         
         c = new TCanvas( Form("canvas%d", countCanvas+1), ctitle, 
                          countCanvas*50+200, countCanvas*20, width, (Int_t)width*0.78 ); 
    
         // set the histogram style
         TMVAGlob::SetSignalAndBackgroundStyle( sig, bgd );
         
         // normalise both signal and background
         TMVAGlob::NormalizeHists( sig, bgd );
         
         // frame limits (choose judicuous x range)
         Float_t nrms = 10;
         cout << "--- Mean and RMS (S): " << sig->GetMean() << ", " << sig->GetRMS() << endl;
         cout << "--- Mean and RMS (B): " << bgd->GetMean() << ", " << bgd->GetRMS() << endl;
         Float_t xmin = TMath::Max( TMath::Min(sig->GetMean() - nrms*sig->GetRMS(), 
                                               bgd->GetMean() - nrms*bgd->GetRMS() ),
                                    sig->GetXaxis()->GetXmin() );
         Float_t xmax = TMath::Min( TMath::Max(sig->GetMean() + nrms*sig->GetRMS(), 
                                               bgd->GetMean() + nrms*bgd->GetRMS() ),
                                    sig->GetXaxis()->GetXmax() );
         Float_t ymin = 0;
         Float_t maxMult = (htype == CompareType) ? 1.3 : 1.2;
         Float_t ymax = TMath::Max( sig->GetMaximum(), bgd->GetMaximum() )*maxMult;
   
         // build a frame
         Int_t nb = 500;
         TString hFrameName(TString("frame") + methodTitle);
         TObject *o = gROOT->FindObject(hFrameName);
         if(o) delete o;
         TH2F* frame = new TH2F( hFrameName, sig->GetTitle(), 
                                 nb, xmin, xmax, nb, ymin, ymax );
         frame->GetXaxis()->SetTitle( methodTitle + ((htype == MVAType || htype == CompareType) ? " response" : "") );
         if      (htype == ProbaType  ) frame->GetXaxis()->SetTitle( "Signal probability" );
         else if (htype == RarityType ) frame->GetXaxis()->SetTitle( "Signal rarity" );
         frame->GetYaxis()->SetTitle("(1/N) dN^{ }/^{ }dx");
         TMVAGlob::SetFrameStyle( frame );
   
         // eventually: draw the frame
         frame->Draw();  
    
         c->GetPad(0)->SetLeftMargin( 0.105 );
         frame->GetYaxis()->SetTitleOffset( 1.2 );

         // Draw legend               
         TLegend *legend= new TLegend( c->GetLeftMargin(), 1 - c->GetTopMargin() - 0.12, 
                                       c->GetLeftMargin() + (htype == CompareType ? 0.40 : 0.3), 1 - c->GetTopMargin() );
         legend->SetFillStyle( 1 );
         legend->AddEntry(sig,TString("Signal")     + ((htype == CompareType) ? " (test sample)" : ""), "F");
         legend->AddEntry(bgd,TString("Background") + ((htype == CompareType) ? " (test sample)" : ""), "F");
         legend->SetBorderSize(1);
         legend->SetMargin( (htype == CompareType ? 0.2 : 0.3) );
         legend->Draw("same");

         // overlay signal and background histograms
         sig->Draw("samehist");
         bgd->Draw("samehist");
   
         if (htype == CompareType) {
            // if overtraining check, load additional histograms
            TH1* sigOv = 0;
            TH1* bgdOv = 0;

            TString ovname = hname += "_Train";
            sigOv = dynamic_cast<TH1*>(titDir->Get( ovname + "_S" ));
            bgdOv = dynamic_cast<TH1*>(titDir->Get( ovname + "_B" ));
      
            if (sigOv == 0 || bgdOv == 0) {
               cout << "+++ Problem in \"mvas.C\": overtraining check histograms do not exist" << endl;
            }
            else {
               cout << "--- Found comparison histograms for overtraining check" << endl;

               TLegend *legend2= new TLegend( 1 - c->GetRightMargin() - 0.42, 1 - c->GetTopMargin() - 0.12,
                                              1 - c->GetRightMargin(), 1 - c->GetTopMargin() );
               legend2->SetFillStyle( 1 );
               legend2->SetBorderSize(1);
               legend2->AddEntry(sigOv,"Signal (training sample)","P");
               legend2->AddEntry(bgdOv,"Background (training sample)","P");
               legend2->SetMargin( 0.1 );
               legend2->Draw("same");
            }
            // normalise both signal and background
            TMVAGlob::NormalizeHists( sigOv, bgdOv );

            Int_t col = sig->GetLineColor();
            sigOv->SetMarkerColor( col );
            sigOv->SetMarkerSize( 0.7 );
            sigOv->SetMarkerStyle( 20 );
            sigOv->SetLineWidth( 1 );
            sigOv->SetLineColor( col );
            sigOv->Draw("e1same");
      
            col = bgd->GetLineColor();
            bgdOv->SetMarkerColor( col );
            bgdOv->SetMarkerSize( 0.7 );
            bgdOv->SetMarkerStyle( 20 );
            bgdOv->SetLineWidth( 1 );
            bgdOv->SetLineColor( col );
            bgdOv->Draw("e1same");

            ymax = TMath::Max( ymax, TMath::Max( sigOv->GetMaximum(), bgdOv->GetMaximum() )*maxMult );
            frame->GetYaxis()->SetLimits( 0, ymax );
      
            // for better visibility, plot thinner lines
            sig->SetLineWidth( 1 );
            bgd->SetLineWidth( 1 );

            // perform K-S test
            cout << "--- Perform Kolmogorov-Smirnov tests" << endl;
            Double_t kolS = sig->KolmogorovTest( sigOv );
            Double_t kolB = bgd->KolmogorovTest( bgdOv );
            cout << "--- Goodness of signal (background) consistency: " << kolS << " (" << kolB << ")" << endl;

            TString probatext = Form( "Kolmogorov-Smirnov test: signal (background) probability = %5.3g (%5.3g)", kolS, kolB );
            TText* tt = new TText( 0.12, 0.74, probatext );
            tt->SetNDC(); tt->SetTextSize( 0.032 ); tt->AppendPad(); 
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
   
         // update canvas
         c->Update();

         // save canvas to file

         TMVAGlob::plot_logo(1.058);
         if (Save_Images) {
            if      (htype == MVAType)     TMVAGlob::imgconv( c, Form("plots/mva_%s",     methodTitle.Data()) );
            else if (htype == ProbaType)   TMVAGlob::imgconv( c, Form("plots/proba_%s",   methodTitle.Data()) ); 
            else if (htype == CompareType) TMVAGlob::imgconv( c, Form("plots/overtrain_%s", methodTitle.Data()) ); 
            else                           TMVAGlob::imgconv( c, Form("plots/rarity_%s",  methodTitle.Data()) ); 
         }
         countCanvas++;
         
      }
      cout << "";
   }
}

