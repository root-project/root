#include "tmvaglob.C"

// this macro plots the resulting MVA distribution (Signal and
// Background overlayed) of different MVA methods run in TMVA
// (e.g. running TMVAnalysis.C).


// input: - Input file (result from TMVA)
//        - use of TMVA plotting TStyle
void mvas( TString fin = "TMVA.root", Bool_t useTMVAStyle = kTRUE )
{
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
   const Int_t noCanvas = 10;
   TCanvas **c = new TCanvas*[noCanvas];
   for (Int_t ic=0; ic<noCanvas; ic++) c[ic] = 0;

   // counter variables
   Int_t countCanvas = 0;
   Int_t countPad    = 1;

   // list of existing MVAs
   const Int_t nveto = 1;
   TString prefix = "MVA_";
   TString suffixSig = "_S";
   TString suffixBgd = "_B";
   TString vetoNames[nveto] = { "muTransform" };

   // search for the right histograms in full list of keys
   TIter next(file->GetListOfKeys());
   TKey *key;
   char fname[200];
   while ((key = (TKey*)next())) {

      // make sure, that we only look at histograms
      TClass *cl = gROOT->GetClass(key->GetClassName());
      if (!cl->InheritsFrom("TH1")) continue;
      TH1 *th1 = (TH1*)key->ReadObj();
      TString hname= th1->GetName();

      if (!hname.BeginsWith( prefix ) || !hname.EndsWith( suffixSig )) continue;

      // check if histogram is vetoed
      Bool_t found = kFALSE;
      for (UInt_t iv=0; iv<nveto; iv++) if (hname.Contains( vetoNames[iv] )) found = kTRUE;
      if (found) continue;

      // remove the signal suffix
      hname.ReplaceAll( suffixSig, "" );

      // retrieve corresponding signal and background histograms   
      TH1* sig = (TH1*)gDirectory->Get( hname + "_S" );
      TH1* bgd = (TH1*)gDirectory->Get( hname + "_B" );
      TH1* all = (TH1*)gDirectory->Get( hname );

      // check that exist
      if (NULL != sig && NULL != bgd) {

         TString hname = sig->GetName();
         cout << "--- Found histogram: " << hname << endl;
      
         // chop off useless stuff
         TString title(sig->GetTitle());
         title.ReplaceAll(prefix, "");
         title.ReplaceAll("_S","");
         sig->SetTitle( TString("MVA output for method: ") + title );

         // create new canvas
         if ((c[countCanvas]==NULL) || (countPad>noPad)) {
            cout << "--- Book canvas no: " << countCanvas << endl;
            char cn[20];
            sprintf( cn, "canvas%d", countCanvas+1 );
            c[countCanvas] = new TCanvas( cn, Form("MVA Output Variables %s",title.Data()), 
                                          countCanvas*50+200, countCanvas*20, width, width*0.78 ); 
            // style
            c[countCanvas]->SetBorderMode(0);
            c[countCanvas]->SetFillColor(10);
        
            c[countCanvas]->Divide(xPad,yPad);
            countPad = 1;
         }       
            
         // set the histogram style
         TMVAGlob::SetSignalAndBackgroundStyle( sig, bgd );

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
         TH2F* frame = new TH2F( TString("frame") + sig->GetName(), sig->GetTitle(), 
                                 nb, xmin, xmax, nb, ymin, ymax );
         frame->GetXaxis()->SetTitle(title);
         frame->GetYaxis()->SetTitle("Normalized");
         TMVAGlob::SetFrameStyle( frame );

         // eventually: draw the frame
         frame->Draw();  

         if (Draw_CFANN_Logy && mvaName[imva] == "CFANN") gPad->SetLogy();

         // overlay signal and background histograms
         sig->Draw("samehist");
         bgd->Draw("samehist");

         // redraw axes
         frame->Draw("sameaxis");

         // Draw legend
         if (countPad==1) {
            TLegend *legend= new TLegend( 0.107, 0.780, 0.400, 0.900  );
            legend->AddEntry(sig,"Signal","F");
            legend->AddEntry(bgd,"Background","F");
            legend->Draw("same");
            legend->SetBorderSize(1);
            legend->SetMargin( 0.3 );
         } 
      
         // save canvas to file
         c[countCanvas]->cd(countPad);
         countPad++;
         if (countPad > noPad) {
            c[countCanvas]->Update();
            TMVAGlob::plot_logo();
            sprintf( fname, "plots/mva_c%i", countCanvas+1 );
            if (Save_Images) TMVAGlob::imgconv( c[countCanvas], &fname[0] );
            countCanvas++;
         }

      }
      // only a single file (stems probably from TMVApplication.cpp)
      else if (NULL != all) {

         TString hname = all->GetName();
         cout << "--- Found (all) histogram: " << hname << endl;
      
         // create new canvas
         if ((c[countCanvas]==NULL) || (countPad>noPad)) {
            cout << "--- Book canvas no: " << countCanvas << endl;
            char cn[20];
            sprintf( cn, "canvas%d", countCanvas+1 );
            c[countCanvas] = new TCanvas( cn, "MVA Output Variables", 
                                          countCanvas*50, countCanvas*20, width, width*0.8 ); 
            // style
            c[countCanvas]->SetBorderMode(0);
            c[countCanvas]->SetFillColor(10);
        
            c[countCanvas]->Divide(xPad,yPad);
            countPad = 1;
         }       
            
         // chop off useless stuff
         TString title(all->GetTitle());
         title.ReplaceAll(prefix, "");
         all->SetTitle( TString("MVA output for method: ") + title );

         // set the histogram style
         TMVAGlob::SetSignalAndBackgroundStyle( sig, bgd, all );

         // frame limits (choose judicuous x range)
         Float_t nrms = 4;
         cout << "--- mean and RMS: " << all->GetMean() << ", " << all->GetRMS() << endl;
         Float_t xmin = TMath::Max( all->GetMean() - nrms*all->GetRMS(),
                                    all->GetXaxis()->GetXmin() );
         Float_t xmax = TMath::Min( all->GetMean() + nrms*all->GetRMS(),
                                    all->GetXaxis()->GetXmax() );
         Float_t ymin = 0;
         Float_t ymax = all->GetMaximum()*1.2 ;

         if (Draw_CFANN_Logy && mvaName[imva] == "CFANN") ymin = 0.01;

         // build a frame
         Int_t nb = 500;
         TH2F* frame = new TH2F( TString("frame") + all->GetName(), all->GetTitle(), 
                                 nb, xmin, xmax, nb, ymin, ymax );
         frame->GetXaxis()->SetTitle(title);
         frame->GetYaxis()->SetTitle("Normalized");
         TMVAGlob::SetFrameStyle( frame );

         // eventually: draw the frame
         frame->Draw();  

         if (Draw_CFANN_Logy && mvaName[imva] == "CFANN") gPad->SetLogy();

         // overlay signal and background histograms
         all->Draw("samehist");

         // redraw axes
         frame->Draw("sameaxis");
      
         // save canvas to file
         c[countCanvas]->cd(countPad);
         countPad++;
         if (countPad > noPad) {
            TMVAGlob::plot_logo();
            c[countCanvas]->Update();
            sprintf( fname, "plots/mva_all_c%i", countCanvas+1 );
            if (Save_Images) TMVAGlob::imgconv( c[countCanvas], &fname[0] );
            countCanvas++;
         }
      }
   }
}
