#include "tmvaglob.C"

void compareanapp( TString finAn = "TMVA.root", TString finApp = "TMVApp.root", bool useTMVAStyle=kTRUE )
{
   // set style and remove existing canvas'
   TMVAGlob::Initialize( useTMVAStyle );

   // switches
   const Bool_t Draw_CFANN_Logy = kFALSE;
   const Bool_t Save_Images     = kTRUE;

   cout << "Reading file: " << finApp << endl;
   TFile *fileApp = new TFile( finApp );

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
   const Int_t nmva = 16;
   TString prefix = "";

   // list of existing MVAs
   const Int_t nmva = 16;
   TString prefix = "";
   TString mvaName[nmva] = { "MVA_Likelihood", 
                             "MVA_LikelihoodD", 
                             "MVA_Fisher", 
                             "MVA_Fisher_Fi", 
                             "MVA_Fisher_Ma", 
                             "MVA_CFMlpANN", 
                             "MVA_TMlpANN", 
                             "MVA_HMatrix", 
                             "MVA_PDERS" , 
                             "MVA_BDT",
                             "MVA_BDTGini",
                             "MVA_BDTMisCl",
                             "MVA_BDTStatSig",
                             "MVA_BDTCE",
                             "MVA_MLP",
                             "MVA_RuleFit" };
   char    fname[200];

   // loop over MVAs
   for (Int_t imva=0; imva<nmva; imva++) {

      // retrieve corresponding signal and background histograms   
      TH1* all = (TH1*)gDirectory->Get( prefix + mvaName[imva] );

      // only a single type (from TMVApplication.cpp)
      if (NULL != all) {

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
         all->SetLineWidth( 2 );
         all->SetLineColor( 1 );

         // normalize 'all'
         all->Scale( 1.0/all->GetSumOfWeights() );

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

         // add the reference histograms         
         //         delete fileApp;
         TFile *fileAn = new TFile( finAn );
         TH1* sig = (TH1*)gDirectory->Get( prefix + mvaName[imva] + "_S" );
         TH1* bgd = (TH1*)gDirectory->Get( prefix + mvaName[imva] + "_B" );
         if (NULL != sig && NULL != bgd) {
            // set the histogram style
            TMVAGlob::SetSignalAndBackgroundStyle( sig, bgd );
            // overlay signal and background histograms
            sig->Scale( 1.0/sig->GetSumOfWeights() );
            bgd->Scale( 1.0/bgd->GetSumOfWeights() );
            sig->Draw("samehist");
            bgd->Draw("samehist");            
         }

         // overlay signal and background histograms
         all->Draw("samehist");

         // reopen TMVApplication output
         TFile *fileApp = new TFile( finApp );

         // redraw axes
         frame->Draw("sameaxis");         
      
         // save canvas to file
         c[countCanvas]->cd(countPad);
         countPad++;
         if (countPad > noPad) {
            TMVAGlob::plot_logo();
            c[countCanvas]->Update();
            sprintf( fname, "plots/compareanapp_c%i", countCanvas+1 );
            if (Save_Images) TMVAGlob::imgconv( c[countCanvas], &fname[0] );
            countCanvas++;
         }
      }
   }
}
