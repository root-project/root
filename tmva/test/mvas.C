#include "tmvaglob.C"

// this macro plots the resulting MVA distributions (Signal and
// Background overlayed) of different MVA methods run in TMVA
// (e.g. running TMVAnalysis.C).

enum HistType { MVAType = 0, ProbaType = 1, RarityType = 2 };

// input: - Input file (result from TMVA)
//        - use of TMVA plotting TStyle
void mvas( TString fin = "TMVA.root", HistType htype = MVAType, Bool_t useTMVAStyle = kTRUE )
{
   // set style and remove existing canvas'
   TMVAGlob::Initialize( useTMVAStyle );

   // switches
   const Bool_t Draw_CFANN_Logy = kFALSE;
   const Bool_t Save_Images     = kTRUE;

   // checks if file with name "fin" is already open, and if not opens one
   TFile* file = TMVAGlob::OpenFile( fin );  

   cout << "--- plotting type: " << htype << endl;

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
   TList methods;
   UInt_t nm = TMVAGlob::GetListOfMethods( methods );
   TIter next(&methods);
   TKey *key, *hkey;   
   while ((key = (TKey*)next())) {

      cout << "--- Found directory: " << ((TDirectory*)key->ReadObj())->GetName() 
           << " --> going in" << endl;

      TString methodName;
      TMVAGlob::GetMethodName(methodName,key);

      cout << "--- Method: " << methodName << endl;

      TDirectory* mDir = (TDirectory*)key->ReadObj();
      TList titles;
      UInt_t ninst = TMVAGlob::GetListOfTitles(mDir,titles);
      TIter nextTitle(&titles);
      TKey *titkey;
      TDirectory *titDir;
      while ((titkey = TMVAGlob::NextKey(nextTitle,"TDirectory"))) {

         titDir = (TDirectory *)titkey->ReadObj();
         TString methodTitle;
         TMVAGlob::GetMethodTitle(methodTitle,titDir);
         TString hname = "MVA_" + methodTitle;
         if      (htype == ProbaType ) hname += "_Proba";
         else if (htype == RarityType) hname += "_Rarity";

         TH1* sig = dynamic_cast<TH1*>(titDir->Get( hname + "_S" ));
         TH1* bgd = dynamic_cast<TH1*>(titDir->Get( hname + "_B" ));

         if(sig==0 || bgd==0) continue;
      
         // chop off useless stuff
         sig->SetTitle( Form("TMVA output for classifier: %s", methodTitle.Data()) );
         if      (htype == ProbaType) 
            sig->SetTitle( Form("TMVA probability for classifier: %s", methodTitle.Data()) );
         else if (htype == RarityType) 
            sig->SetTitle( Form("TMVA Rarity for classifier: %s", methodTitle.Data()) );
         
         // create new canvas
         TString ctitle = ((htype == MVAType) ? 
                           Form("TMVA output %s",methodTitle.Data()) : 
                           (htype == ProbaType) ? 
                           Form("TMVA probability %s",methodTitle.Data()) :
                           Form("TMVA rarity %s",methodTitle.Data()));

         TString cname = ((htype == MVAType) ? 
                          Form("output_%s",methodTitle.Data()) : 
                          (htype == ProbaType) ? 
                          Form("probability_%s",methodTitle.Data()) :
                          Form("rarity_%s",methodTitle.Data()));

         c = new TCanvas( Form("canvas%d", countCanvas+1), ctitle, 
                          countCanvas*50+200, countCanvas*20, width, width*0.78 ); 
          
         // set the histogram style
         TMVAGlob::SetSignalAndBackgroundStyle( sig, bgd );
         
         // normalise both signal and background
         TMVAGlob::NormalizeHists( sig, bgd );
         
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
         Float_t ymax = TMath::Max( sig->GetMaximum(), bgd->GetMaximum() )*1.2 ;
         
         if (Draw_CFANN_Logy && mvaName[imva] == "CFANN") ymin = 0.01;
         
         // build a frame
         Int_t nb = 500;
         TH2F* frame = new TH2F( TString("frame") + methodTitle, sig->GetTitle(), 
                                 nb, xmin, xmax, nb, ymin, ymax );
         frame->GetXaxis()->SetTitle(methodTitle);
         if      (htype == ProbaType ) frame->GetXaxis()->SetTitle( "Signal probability" );
         else if (htype == RarityType) frame->GetXaxis()->SetTitle( "Signal rarity" );
         frame->GetYaxis()->SetTitle("Normalized");
         TMVAGlob::SetFrameStyle( frame );
         
         // eventually: draw the frame
         frame->Draw();  
         
         c->GetPad(0)->SetLeftMargin( 0.105 );
         frame->GetYaxis()->SetTitleOffset( 1.2 );
         
         if (Draw_CFANN_Logy && mvaName[imva] == "CFANN") c->SetLogy();
         
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
         
         // redraw axes
         frame->Draw("sameaxis");
         
         // text for overflows
         Int_t nbin = sig->GetNbinsX();
         TString uoflow = Form( "U/O-flow (S,B): (%.1f, %.1f)% / (%.1f, %.1f)%", 
                                sig->GetBinContent(0)*100, bgd->GetBinContent(0)*100,
                                sig->GetBinContent(nbin+1)*100, bgd->GetBinContent(nbin+1)*100 );
         TText* t = new TText( 0.975, 0.115, uoflow );
         t->SetNDC();
         t->SetTextSize( 0.030 );
         t->SetTextAngle( 90 );
         t->AppendPad();    
         
         // save canvas to file
         c->Update();
         TMVAGlob::plot_logo();
         if (Save_Images) {
            if      (htype == MVAType)   TMVAGlob::imgconv( c, Form("plots/mva_%s",    methodTitle.Data()) );
            else if (htype == ProbaType) TMVAGlob::imgconv( c, Form("plots/proba_%s",  methodTitle.Data()) ); 
            else                         TMVAGlob::imgconv( c, Form("plots/rarity_%s", methodTitle.Data()) ); 
         }
         countCanvas++;
      }
   }
}

