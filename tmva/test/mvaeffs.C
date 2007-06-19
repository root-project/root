#include "tmvaglob.C"

// this macro plots the signal and background efficiencies
// as a function of the MVA cut.

enum PlotType { EffPurity = 0 };

// input: - Input file (result from TMVA)
//        - use of TMVA plotting TStyle
void mvaeffs( TString fin = "TMVA.root", PlotType ptype = EffPurity, Bool_t useTMVAStyle = kTRUE )
{
   // set style and remove existing canvas'
   TMVAGlob::Initialize( useTMVAStyle );

   // switches
   const Bool_t Save_Images = kTRUE;

   // checks if file with name "fin" is already open, and if not opens one
   TFile* file = TMVAGlob::OpenFile( fin );  

   cout << "--- plotting type: " << ptype << endl;

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
   TKey *key, *hkey;   
   while ((key = (TKey*)next())) {

      if (TString(key->GetClassName()) != "TDirectory" && TString(key->GetClassName()) != "TDirectoryFile") continue;
      if (!TString(key->GetName()).BeginsWith("Method_")) continue;

      cout << "--- Found directory: " << ((TDirectory*)key->ReadObj())->GetName() 
           << " --> going in" << endl;
      TString methodName;
      TMVAGlob::GetMethodName(methodName,key);

      TDirectory* mDir = (TDirectory*)key->ReadObj();
      //
      TList titles;
      UInt_t ntit = TMVAGlob::GetListOfTitles(mDir,titles);
      TIter nextTitle(&titles);
      TKey *titkey;
      TDirectory *titDir;
      while ((titkey = TMVAGlob::NextKey(nextTitle,"TDirectory"))) {
         titDir = (TDirectory *)titkey->ReadObj();
         TString methodTitle;
         TMVAGlob::GetMethodTitle(methodTitle,titDir);

         TString hname = "MVA_" + methodTitle;

         cout << "--- Classifier: " << methodTitle << endl;

         TH1* sig = dynamic_cast<TH1*>(titDir->Get( hname + "_S" ));
         TH1* bgd = dynamic_cast<TH1*>(titDir->Get( hname + "_B" ));

         TH1* sigE = dynamic_cast<TH1*>(titDir->Get( hname + "_effS" ));
         TH1* bgdE = dynamic_cast<TH1*>(titDir->Get( hname + "_effB" ));      

         if (sigE==0 || bgdE==0) continue;
      
         // compute signal purity and quality (=eff*purity)
         // delete if existing

         TString pname  = Form( "purS_%s", methodTitle.Data() );
         TString epname = Form( "effpurS_%s", methodTitle.Data() );
         TH1* purS = new TH1F( pname, pname, 
                               sigE->GetNbinsX(), sigE->GetXaxis()->GetXmin(), sigE->GetXaxis()->GetXmax() );
         TH1* effpurS = new TH1F( epname, epname,
                                  sigE->GetNbinsX(), sigE->GetXaxis()->GetXmin(), sigE->GetXaxis()->GetXmax() );

         for (Int_t i=1; i<=sigE->GetNbinsX(); i++) {
            Float_t eS = sigE->GetBinContent( i );
            Float_t eB = bgdE->GetBinContent( i );
            Float_t  eps = 1e-20;
            if (eS < eps) eS = eps;
            if (eB < eps) eB = eps;
            purS->SetBinContent( i, eS/(eS+eB) );
            effpurS->SetBinContent( i, eS*purS->GetBinContent( i ) );
         }
         
         // chop off useless stuff
         sigE->SetTitle( Form("Cut efficiencies for %s classifier", methodTitle.Data()) );
         
         // create new canvas
         TString cname = Form("Cut efficiencies for %s classifier",methodTitle.Data());
         
         c = new TCanvas( Form("canvas%d", countCanvas+1), cname, 
                          countCanvas*50+200, countCanvas*20, width, width*0.78 ); 
         
         // draw grid
         c->SetGrid(1);
         
         // set the histogram style
         TMVAGlob::SetSignalAndBackgroundStyle( sigE, bgdE );
         TMVAGlob::SetSignalAndBackgroundStyle( purS, bgdE );
         TMVAGlob::SetSignalAndBackgroundStyle( effpurS, bgdE );
         sigE->SetFillStyle( 0 );
         bgdE->SetFillStyle( 0 );
         sigE->SetLineWidth( 3 );
         bgdE->SetLineWidth( 3 );
         
         // the purity and quality
         TStyle *TMVAStyle = gROOT->GetStyle("Plain"); // our style is based on Plain
         TMVAStyle->SetLineStyleString( 5, "[32 22]" );
         TMVAStyle->SetLineStyleString( 6, "[12 22]" );
         purS->SetFillStyle( 0 );
         purS->SetLineWidth( 2 );
         purS->SetLineStyle( 5 );
         effpurS->SetFillStyle( 0 );
         effpurS->SetLineWidth( 2 );
         effpurS->SetLineStyle( 6 );
         
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
         Float_t ymax = 1.25;
         
         // build a frame
         Int_t nb = 500;
         TH2F* frame = new TH2F( TString("frame") + sigE->GetName(), sigE->GetTitle(), 
                                 nb, xmin, xmax, nb, ymin, ymax );
         frame->GetXaxis()->SetTitle( Form( "Cut on %s output", methodTitle.Data() ) );
         frame->GetYaxis()->SetTitle("Efficiency");
         TMVAGlob::SetFrameStyle( frame );
         
         // draw the frame
         frame->Draw();  
         
         // overlay signal and background histograms
         sigE->Draw("samehistl");
         bgdE->Draw("samehistl");
         
         // and the signal purity and quality
         purS->Draw("samehistl");
         effpurS->Draw("samehistl");
         
         // redraw axes
         frame->Draw("sameaxis");
         
         // add line to indicate eff=1
         TLine* line = new TLine( xmin, 1, xmax, 1 );
         line->SetLineWidth(1);
         line->SetLineStyle(1);
         line->Draw();
                    
         // Draw legend               
         TLegend *legend1= new TLegend( c->GetLeftMargin(), 1 - c->GetTopMargin() - 0.12, 
                                        c->GetLeftMargin() + 0.4, 1 - c->GetTopMargin() );
         legend1->SetFillStyle( 1 );
         legend1->AddEntry(sigE,"Signal efficiency","L");
         legend1->AddEntry(bgdE,"Background efficiency","L");
         legend1->Draw("same");
         legend1->SetBorderSize(1);
         legend1->SetMargin( 0.3 );

         TLegend *legend2= new TLegend( 1 - c->GetRightMargin() - 0.42, 1 - c->GetTopMargin() - 0.12, 
                                        1 - c->GetRightMargin(), 1 - c->GetTopMargin() );
         legend2->SetFillStyle( 1 );
         legend2->AddEntry(purS,"Signal purity","L");
         legend2->AddEntry(effpurS,"Signal efficiency*purity","L");
         legend2->Draw("same");
         legend2->SetBorderSize(1);
         legend2->SetMargin( 0.3 );
         
         // print comments
         TText* t1 = new TText( 0.13, 0.18, "The purity curves use an equal number of" );
         TText* t2 = new TText( 0.13, 0.18-0.03, "signal and background events before cutting" );
         t1->SetNDC();
         t1->SetTextSize( 0.026 );
         t1->AppendPad();    
         t2->SetNDC();
         t2->SetTextSize( 0.026 );
         t2->AppendPad();    
         
         // save canvas to file
         c->Update();
         //         TMVAGlob::plot_logo();
         if (Save_Images) {
            TMVAGlob::imgconv( c, Form("plots/mvaeffs_%s", methodTitle.Data()) ); 
         }
         countCanvas++;
      }
   }
}

